# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Generating PTX that makes ptxas show us the instruction set.

The harvester's job is to get the vendor compiler to emit as much of the ISA as
it can be persuaded to emit, with unambiguous attribution from each SASS
instruction back to the PTX that caused it.

Two design decisions do most of the work:

*One instruction under test per kernel.* The kernel loads its operands from
global memory, performs exactly one operation, and stores the result. Anything
else and we would be guessing which SASS line came from which PTX line.

*Operands that cannot be folded.* Values are read from memory rather than
written as literals, so ptxas cannot constant-fold the operation away, and the
result is stored so it cannot be dead-code eliminated. Getting this wrong is the
classic way to spend a day harvesting an empty corpus.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

__all__ = ["PTX_VERSION", "Snippet", "generate"]

PTX_VERSION = "9.0"

# PTX type -> (register class prefix, bytes, ptx register declaration type)
_TYPES: dict[str, tuple[str, int, str]] = {
    "s16": ("h", 2, "b16"),
    "u16": ("h", 2, "b16"),
    "s32": ("r", 4, "b32"),
    "u32": ("r", 4, "b32"),
    "s64": ("d", 8, "b64"),
    "u64": ("d", 8, "b64"),
    "f16": ("h", 2, "b16"),
    "f32": ("f", 4, "f32"),
    "f64": ("g", 8, "f64"),
    "b16": ("h", 2, "b16"),
    "b32": ("r", 4, "b32"),
    "b64": ("d", 8, "b64"),
}

_INT = ("s16", "u16", "s32", "u32", "s64", "u64")
_INT32 = ("s32", "u32")
_FLOAT = ("f32", "f64")
_BITS = ("b16", "b32", "b64")


@dataclass(frozen=True, slots=True)
class Snippet:
    """One generated kernel: a name, its PTX, and what we expect it to exercise."""

    name: str
    ptx: str
    family: str
    op: str
    ptx_type: str
    modifiers: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        mods = "".join("." + m for m in self.modifiers)
        return f"{self.op}{mods}.{self.ptx_type}"


def _kernel(name: str, body: str, decls: str) -> str:
    return f""".version {PTX_VERSION}
.target sm_120a
.address_size 64

.visible .entry {name}(.param .u64 pin, .param .u64 pout)
{{
    .reg .pred  %p<4>;
    .reg .b16   %h<8>;
    .reg .b32   %r<8>;
    .reg .b64   %d<8>;
    .reg .f32   %f<8>;
    .reg .f64   %g<8>;
    .reg .b64   %in, %out;
{decls}
    ld.param.u64  %in, [pin];
    cvta.to.global.u64 %in, %in;
    ld.param.u64  %out, [pout];
    cvta.to.global.u64 %out, %out;
{body}
    ret;
}}
"""


def _reg(ptx_type: str, idx: int) -> str:
    prefix, _, _ = _TYPES[ptx_type]
    return f"%{prefix}{idx}"


# `ld` and `st` have no `.f16`: a half moves as raw bits. emitting one made all
# 25 half-precision kernels fail to build, silently, from the start
_MOVE_TYPE = {"f16": "b16"}


def _load(ptx_type: str, dst: str, off: int) -> str:
    return f"    ld.global.{_MOVE_TYPE.get(ptx_type, ptx_type)} {dst}, [%in+{off}];"


def _store(ptx_type: str, src: str) -> str:
    return f"    st.global.{_MOVE_TYPE.get(ptx_type, ptx_type)} [%out], {src};"


def _binary(op: str, ptx_type: str, mods: tuple[str, ...] = ()) -> Snippet:
    """dst = op(a, b) for a two-source instruction."""
    _, size, _ = _TYPES[ptx_type]
    a, b, d = _reg(ptx_type, 1), _reg(ptx_type, 2), _reg(ptx_type, 3)
    suffix = "".join("." + m for m in mods)
    body = "\n".join(
        [
            _load(ptx_type, a, 0),
            _load(ptx_type, b, size),
            f"    {op}{suffix}.{ptx_type} {d}, {a}, {b};",
            _store(ptx_type, d),
        ]
    )
    name = f"k_{op}{''.join('_' + m for m in mods)}_{ptx_type}"
    return Snippet(name, _kernel(name, body, ""), "binary", op, ptx_type, mods)


# a float literal has to be given as its bits, and the unsigned types are kept
# positive so the value means the same thing it does in the signed ones
_IMMEDIATE = {
    "s16": "-24",
    "u16": "24",
    "s32": "-24",
    "u32": "24",
    "s64": "-24",
    "u64": "24",
    "f16": "0hCE00",
    "f32": "0fC1C00000",
    "f64": "0dC038000000000000",
    "b16": "0x1234",
    "b32": "0x12345678",
    "b64": "0x123456789abcdef0",
}


def _binary_imm(op: str, ptx_type: str, mods: tuple[str, ...] = ()) -> Snippet:
    """dst = op(a, immediate), which is a different encoding from op(a, b).

    Everything else here is register to register, so the immediate form of every
    mnemonic went unharvested and the assembler had nothing to write `FADD R0,
    R1, -24` with. The operand shape already tells the two apart; this is what
    makes the second one exist.
    """
    a, d = _reg(ptx_type, 1), _reg(ptx_type, 3)
    suffix = "".join("." + m for m in mods)
    body = "\n".join(
        [
            _load(ptx_type, a, 0),
            f"    {op}{suffix}.{ptx_type} {d}, {a}, {_IMMEDIATE[ptx_type]};",
            _store(ptx_type, d),
        ]
    )
    name = f"k_{op}{''.join('_' + m for m in mods)}_imm_{ptx_type}"
    return Snippet(name, _kernel(name, body, ""), "binary-imm", op, ptx_type, mods)


def _widening(op: str, ptx_type: str) -> Snippet:
    """mul.wide and mad.wide produce a result twice the operand width.

    Using one type for all three operands, which is right for every other
    binary form, silently produces a corpus entry ptxas always rejects.
    """
    wide = {"s16": "s32", "u16": "u32", "s32": "s64", "u32": "u64"}[ptx_type]
    _, size, _ = _TYPES[ptx_type]
    a, b, d = _reg(ptx_type, 1), _reg(ptx_type, 2), _reg(wide, 5)
    body = "\n".join(
        [
            _load(ptx_type, a, 0),
            _load(ptx_type, b, size),
            f"    {op}.wide.{ptx_type} {d}, {a}, {b};",
            _store(wide, d),
        ]
    )
    name = f"k_{op}_wide_{ptx_type}"
    return Snippet(name, _kernel(name, body, ""), "widening", op, ptx_type, ("wide",))


def _ternary(op: str, ptx_type: str, mods: tuple[str, ...] = ()) -> Snippet:
    _, size, _ = _TYPES[ptx_type]
    a, b, c, d = (_reg(ptx_type, i) for i in (1, 2, 3, 4))
    suffix = "".join("." + m for m in mods)
    body = "\n".join(
        [
            _load(ptx_type, a, 0),
            _load(ptx_type, b, size),
            _load(ptx_type, c, 2 * size),
            f"    {op}{suffix}.{ptx_type} {d}, {a}, {b}, {c};",
            _store(ptx_type, d),
        ]
    )
    name = f"k_{op}{''.join('_' + m for m in mods)}_{ptx_type}"
    return Snippet(name, _kernel(name, body, ""), "ternary", op, ptx_type, mods)


def _unary(
    op: str, ptx_type: str, mods: tuple[str, ...] = (), result_type: str | None = None
) -> Snippet:
    # the result is not always the operand's width: `popc.b64` counts a 64-bit
    # word into a 32-bit one
    out_type = result_type or ptx_type
    a, d = _reg(ptx_type, 1), _reg(out_type, 2)
    suffix = "".join("." + m for m in mods)
    body = "\n".join(
        [
            _load(ptx_type, a, 0),
            f"    {op}{suffix}.{ptx_type} {d}, {a};",
            _store(out_type, d),
        ]
    )
    name = f"k_{op}{''.join('_' + m for m in mods)}_{ptx_type}"
    return Snippet(name, _kernel(name, body, ""), "unary", op, ptx_type, mods)


def _compare(op: str, cmp_op: str, ptx_type: str) -> Snippet:
    """setp writes a predicate; select it back into a storable value."""
    _, size, _ = _TYPES[ptx_type]
    a, b = _reg(ptx_type, 1), _reg(ptx_type, 2)
    body = "\n".join(
        [
            _load(ptx_type, a, 0),
            _load(ptx_type, b, size),
            f"    {op}.{cmp_op}.{ptx_type} %p1, {a}, {b};",
            "    selp.b32 %r7, 1, 0, %p1;",
            "    st.global.b32 [%out], %r7;",
        ]
    )
    name = f"k_{op}_{cmp_op}_{ptx_type}"
    return Snippet(name, _kernel(name, body, ""), "compare", op, ptx_type, (cmp_op,))


def _convert(dst_type: str, src_type: str, mods: tuple[str, ...] = ()) -> Snippet:
    a, d = _reg(src_type, 1), _reg(dst_type, 2)
    suffix = "".join("." + m for m in mods)
    body = "\n".join(
        [
            _load(src_type, a, 0),
            f"    cvt{suffix}.{dst_type}.{src_type} {d}, {a};",
            _store(dst_type, d),
        ]
    )
    name = f"k_cvt{''.join('_' + m for m in mods)}_{dst_type}_{src_type}"
    return Snippet(name, _kernel(name, body, ""), "convert", "cvt", f"{dst_type}.{src_type}", mods)


def _shift(op: str, ptx_type: str) -> Snippet:
    a, d = _reg(ptx_type, 1), _reg(ptx_type, 3)
    body = "\n".join(
        [
            _load(ptx_type, a, 0),
            "    ld.global.u32 %r6, [%in+16];",
            f"    {op}.{ptx_type} {d}, {a}, %r6;",
            _store(ptx_type, d),
        ]
    )
    name = f"k_{op}_{ptx_type}"
    return Snippet(name, _kernel(name, body, ""), "shift", op, ptx_type)


def _atomic(op: str, ptx_type: str, space: str = "global") -> Snippet:
    a, d = _reg(ptx_type, 1), _reg(ptx_type, 3)
    body = "\n".join(
        [
            _load(ptx_type, a, 0),
            f"    atom.{space}.{op}.{ptx_type} {d}, [%out], {a};",
            _store(ptx_type, d),
        ]
    )
    name = f"k_atom_{space}_{op}_{ptx_type}"
    return Snippet(name, _kernel(name, body, ""), "atomic", f"atom.{op}", ptx_type, (space,))


def _atomic_shared(op: str, ptx_type: str) -> Snippet:
    """A shared-memory atomic, which is `ATOMS` rather than `ATOMG`.

    `_atomic` has carried a `space` parameter since it was written and was never
    passed anything but its default, so the shared opcode was never harvested.

    One slot per thread, initialised before a barrier. Sharing one slot makes the
    returned prior value depend on which thread got there first, so the kernel
    answers differently run to run and the round trip can only exclude it. Here
    every thread reads back what it wrote, so the value is fixed while the
    dependency on the atomic's result, which is the thing being scheduled, is not.
    """
    a, d = _reg(ptx_type, 1), _reg(ptx_type, 3)
    body = "\n".join(
        [
            _load(ptx_type, a, 0),
            "    mov.u32 %r6, %tid.x;",
            "    shl.b32 %r6, %r6, 2;",
            "    mov.u32 %r7, slot;",
            "    add.u32 %r7, %r7, %r6;",
            f"    st.shared.{_MOVE_TYPE.get(ptx_type, ptx_type)} [%r7], {a};",
            "    bar.sync 0;",
            f"    atom.shared.{op}.{ptx_type} {d}, [%r7], {a};",
            _store(ptx_type, d),
        ]
    )
    name = f"k_atom_shared_{op}_{ptx_type}"
    decls = "    .shared .align 8 .b8 slot[256];"
    return Snippet(name, _kernel(name, body, decls), "atomic", f"atom.{op}", ptx_type, ("shared",))


def _reduce(op: str, ptx_type: str, space: str = "global") -> Snippet:
    """`red` is an atomic that returns nothing, and a separate opcode for it."""
    a = _reg(ptx_type, 1)
    if space == "shared":
        body = "\n".join(
            [
                _load(ptx_type, a, 0),
                "    mov.u32 %r7, slot;",
                # initialised first: reducing into whatever the last kernel left
                # in shared memory answers differently every run
                f"    st.shared.{ptx_type} [%r7], 0;",
                "    bar.sync 0;",
                f"    red.shared.{op}.{ptx_type} [%r7], {a};",
                "    bar.sync 0;",
                f"    ld.shared.{ptx_type} {a}, [%r7];",
                _store(ptx_type, a),
            ]
        )
        decls = "    .shared .align 8 .b8 slot[16];"
    else:
        body = "\n".join([_load(ptx_type, a, 0), f"    red.global.{op}.{ptx_type} [%out], {a};"])
        decls = ""
    name = f"k_red_{space}_{op}_{ptx_type}"
    return Snippet(name, _kernel(name, body, decls), "atomic", f"red.{op}", ptx_type, (space,))


def _memory(op: str, ptx_type: str, space: str, mods: tuple[str, ...] = ()) -> Snippet:
    """Exercise a load/store space and cache-modifier combination."""
    d = _reg(ptx_type, 1)
    suffix = "".join("." + m for m in mods)
    if op == "ld":
        body = "\n".join([f"    ld.{space}{suffix}.{ptx_type} {d}, [%in];", _store(ptx_type, d)])
    else:
        body = "\n".join([_load(ptx_type, d, 0), f"    st.{space}{suffix}.{ptx_type} [%out], {d};"])
    name = f"k_{op}_{space}{''.join('_' + m for m in mods)}_{ptx_type}"
    return Snippet(name, _kernel(name, body, ""), "memory", op, ptx_type, (space, *mods))


def _async_copy(kind: str, size: int) -> Snippet:
    """`cp.async`, which is `LDGSTS` and a third way of tracking completion.

    Not a scoreboard on the copy and not a stall count: `LDGDEPBAR` signals, and
    `DEPBAR.LE SB0, 0x0` waits for the outstanding copies to drain. The
    scoreboard is named in `DEPBAR`'s operand text rather than in its control
    word, so renumbering the signaller silently unpairs them, which is why these
    kernels were out of the corpus until the scheduler learned to pin it
    (finding 27).
    """
    body = "\n".join(
        [
            "    mov.u32 %r1, buf;",
            "    mov.u32 %r2, %tid.x;",
            f"    mul.lo.u32 %r3, %r2, {size};",
            "    add.u32 %r1, %r1, %r3;",
            f"    cp.async.{kind}.shared.global [%r1], [%in], {size};",
            "    cp.async.commit_group;",
            "    cp.async.wait_group 0;",
            "    bar.sync 0;",
            "    ld.shared.b32 %r5, [%r1];",
            "    st.global.b32 [%out], %r5;",
        ]
    )
    name = f"k_cp_async_{kind}_{size}"
    decls = f"    .shared .align 16 .b8 buf[{size * 32}];"
    return Snippet(name, _kernel(name, body, decls), "async", "cp.async", "b32", (kind, str(size)))


def _window(ptx_type: str, space: str) -> Snippet:
    """Store into a window of `space` and read the same address back.

    Shared and local memory are their own address spaces, so the global pointer
    the harness passes in is not an address in either of them. Handing one to
    `ld.shared` compiles and faults, which is a kernel that can be harvested and
    never run, and both spaces were exactly that here until this replaced them.

    One slot per thread, so the value read back is the one this thread wrote
    whatever order the warp runs in, and the barrier makes that hold across the
    block as well as within a thread.
    """
    move = _MOVE_TYPE.get(ptx_type, ptx_type)
    width = _TYPES[ptx_type][1]
    body = [
        _load(ptx_type, _reg(ptx_type, 1), 0),
        "    mov.u32 %r6, %tid.x;",
        f"    mul.lo.u32 %r6, %r6, {width};",
        "    mov.u32 %r7, slot;",
        "    add.u32 %r7, %r7, %r6;",
        f"    st.{space}.{move} [%r7], {_reg(ptx_type, 1)};",
    ]
    if space == "shared":
        # local memory is per-thread, so only shared needs the block to agree
        body.append("    bar.sync 0;")
    body += [
        f"    ld.{space}.{move} {_reg(ptx_type, 2)}, [%r7];",
        _store(ptx_type, _reg(ptx_type, 2)),
    ]
    name = f"k_ldst_{space}_{ptx_type}"
    decls = f"    .{space} .align 8 .b8 slot[{width * 32}];"
    return Snippet(
        name, _kernel(name, "\n".join(body), decls), "memory", "ldst", ptx_type, (space,)
    )


def _special(name_suffix: str, body: str, op: str) -> Snippet:
    name = f"k_special_{name_suffix}"
    return Snippet(name, _kernel(name, body, ""), "special", op, "b32")


def generate() -> list[Snippet]:
    """The full generated corpus.

    Breadth over depth on purpose. A form that ptxas rejects costs one failed
    subprocess and is recorded as a negative result; a form we never tried is
    invisible, and invisible gaps are what make an ISA database quietly wrong.
    """
    out: list[Snippet] = []

    # integer and float arithmetic
    for t in _INT:
        out += [_binary(op, t) for op in ("add", "sub", "min", "max")]
        out.append(_binary("mul", t, ("lo",)))
        out.append(_binary("mul", t, ("hi",)))
        if t in ("s16", "u16", "s32", "u32"):
            out.append(_widening("mul", t))
    for t in _INT32:
        out += [_ternary("mad", t, ("lo",)), _ternary("mad", t, ("hi",))]
        out += [_binary("div", t), _binary("rem", t)]
    out += [_binary("add", "s32", ("sat",))]

    for t in _FLOAT:
        out += [_binary(op, t) for op in ("add", "sub", "mul", "min", "max")]
        # float division has no default rounding mode in PTX; it has to be named
        out.append(_binary("div", t, ("rn",)))
        out += [_ternary("fma", t, ("rn",))]
        out += [_unary(op, t) for op in ("abs", "neg")]
    out += [
        _binary("add", "f32", ("rn",)),
        _binary("add", "f32", ("rz",)),
        _binary("add", "f32", ("rm",)),
        _binary("add", "f32", ("rp",)),
    ]
    out += [_binary("add", "f32", ("ftz",)), _binary("mul", "f32", ("sat",))]

    # the same arithmetic with a literal source. everything above is register to
    # register, which left the immediate encoding of each of these unharvested
    for t in _INT:
        out += [_binary_imm(op, t) for op in ("add", "sub", "min", "max")]
        out.append(_binary_imm("mul", t, ("lo",)))
    for t in _FLOAT:
        out += [_binary_imm(op, t) for op in ("add", "sub", "mul", "min", "max")]
    out.append(_binary_imm("add", "f32", ("ftz",)))
    # no f16 row: `add.f16` takes no immediate operand, in either literal syntax
    for t in _BITS:
        out += [_binary_imm(op, t) for op in ("and", "or", "xor")]
    out += [
        _unary(op, "f32", ("approx",))
        for op in ("rcp", "sqrt", "rsqrt", "sin", "cos", "lg2", "ex2")
    ]
    out += [_unary("sqrt", "f32", ("rn",)), _unary("rcp", "f64", ("rn",))]

    # half precision, packed and scalar
    for op in ("add", "sub", "mul", "min", "max"):
        out.append(_binary(op, "f16"))
    out.append(_ternary("fma", "f16", ("rn",)))

    # bitwise, shift, bitfield
    for t in _BITS:
        out += [_binary(op, t) for op in ("and", "or", "xor")]
        out.append(_unary("not", t))
    for t in ("b32", "b64"):
        out += [_shift("shl", t), _shift("shr", t)]
    out += [_unary("popc", "b32"), _unary("popc", "b64", result_type="b32")]
    out += [_unary("clz", "b32"), _unary("clz", "b64", result_type="b32")]
    out += [_unary("brev", "b32"), _unary("brev", "b64")]

    # comparison across every predicate ptxas accepts for the type class
    for t in ("s32", "u32", "s64", "f32", "f64"):
        cmps = (
            ("eq", "ne", "lt", "le", "gt", "ge")
            if not t.startswith("f")
            else ("eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "num", "nan")
        )
        out += [_compare("setp", c, t) for c in cmps]

    # conversions, which is where a large slice of the opcode space lives
    conv_types = ("s16", "u16", "s32", "u32", "s64", "u64", "f16", "f32", "f64")
    for dst, src in itertools.permutations(conv_types, 2):
        mods: tuple[str, ...] = ()
        if dst.startswith("f") and src.startswith("f"):
            # a widening float conversion is exact and PTX rejects a rounding
            # modifier on one; only a narrowing one has to say how to round
            mods = ("rn",) if _TYPES[dst][1] < _TYPES[src][1] else ()
        elif dst.startswith(("s", "u")) and src.startswith("f"):
            mods = ("rni",)
        elif dst.startswith("f") and src.startswith(("s", "u")):
            mods = ("rn",)
        out.append(_convert(dst, src, mods))

    # memory spaces and cache modifiers
    for t in ("b32", "b64", "u16"):
        out.append(_memory("ld", t, "global"))
        out.append(_memory("st", t, "global"))
        out.append(_memory("ld", t, "const"))
        # shared and local go through a slot they actually own, so they run
        out.append(_window(t, "shared"))
        out.append(_window(t, "local"))
    for mod in ("ca", "cg", "cs", "lu", "cv"):
        out.append(_memory("ld", "b32", "global", (mod,)))
    for mod in ("wb", "cg", "cs", "wt"):
        out.append(_memory("st", "b32", "global", (mod,)))
    for t in ("b32", "b64", "b128"):
        if t == "b128":
            continue
        out.append(_memory("ld", t, "global", ("nc",)))

    # atomics and reductions
    for op in ("add", "min", "max"):
        for t in ("u32", "s32", "u64"):
            out.append(_atomic(op, t))
    # `.and`, `.or`, `.xor` and `.exch` are bit operations and PTX types them as
    # such: only `.b32` and `.b64` are legal, whatever the value is meant to be
    for op in ("and", "or", "xor", "exch"):
        for t in ("b32", "b64"):
            out.append(_atomic(op, t))
    out.append(_atomic("add", "f32"))
    out.append(_atomic("add", "f64"))
    # the shared-space atomic is a different opcode from the global one
    for op in ("add", "min", "max", "exch"):
        out.append(_atomic_shared(op, "u32" if op != "exch" else "b32"))
    # a reduction is an atomic that returns nothing, and is its own opcode again
    for op in ("add", "min", "max", "and", "or", "xor"):
        out.append(_reduce(op, "b32" if op in ("and", "or", "xor") else "u32"))
    out.append(_reduce("add", "u32", "shared"))
    for size in (4, 8, 16):
        for kind in ("ca", "cg"):
            if kind == "cg" and size != 16:
                continue  # `cg` bypasses L1, which only the widest copy may do
            out.append(_async_copy(kind, size))

    # warp-level and special registers, which have no arithmetic equivalent
    out += [
        _special("tid", "    mov.u32 %r1, %tid.x;\n    st.global.b32 [%out], %r1;", "mov.sreg"),
        _special("ctaid", "    mov.u32 %r1, %ctaid.x;\n    st.global.b32 [%out], %r1;", "mov.sreg"),
        _special("laneid", "    mov.u32 %r1, %laneid;\n    st.global.b32 [%out], %r1;", "mov.sreg"),
        _special("smid", "    mov.u32 %r1, %smid;\n    st.global.b32 [%out], %r1;", "mov.sreg"),
        _special("clock", "    mov.u32 %r1, %clock;\n    st.global.b32 [%out], %r1;", "mov.sreg"),
        _special("gridid", "    mov.u64 %d1, %gridid;\n    st.global.b64 [%out], %d1;", "mov.sreg"),
        _special(
            "shfl",
            "    ld.global.b32 %r1, [%in];\n"
            "    shfl.sync.bfly.b32 %r2|%p1, %r1, 16, 31, -1;\n"
            "    st.global.b32 [%out], %r2;",
            "shfl.sync",
        ),
        _special(
            "vote",
            "    ld.global.b32 %r1, [%in];\n"
            "    setp.gt.s32 %p1, %r1, 0;\n"
            "    vote.sync.ballot.b32 %r2, %p1, -1;\n"
            "    st.global.b32 [%out], %r2;",
            "vote.sync",
        ),
        _special(
            "match",
            "    ld.global.b32 %r1, [%in];\n"
            "    match.any.sync.b32 %r2, %r1, -1;\n"
            "    st.global.b32 [%out], %r2;",
            "match.sync",
        ),
        _special(
            "activemask",
            "    activemask.b32 %r1;\n    st.global.b32 [%out], %r1;",
            "activemask",
        ),
        _special(
            "bar",
            "    ld.global.b32 %r1, [%in];\n    bar.sync 0;\n    st.global.b32 [%out], %r1;",
            "bar.sync",
        ),
        _special(
            "membar",
            "    ld.global.b32 %r1, [%in];\n    membar.gl;\n    st.global.b32 [%out], %r1;",
            "membar",
        ),
        _special(
            "prmt",
            "    ld.global.b32 %r1, [%in];\n"
            "    ld.global.b32 %r2, [%in+4];\n"
            "    prmt.b32 %r3, %r1, %r2, 0x4321;\n"
            "    st.global.b32 [%out], %r3;",
            "prmt",
        ),
        _special(
            "bfe",
            "    ld.global.b32 %r1, [%in];\n"
            "    bfe.u32 %r2, %r1, 4, 8;\n"
            "    st.global.b32 [%out], %r2;",
            "bfe",
        ),
        _special(
            "bfi",
            "    ld.global.b32 %r1, [%in];\n"
            "    ld.global.b32 %r2, [%in+4];\n"
            "    bfi.b32 %r3, %r1, %r2, 4, 8;\n"
            "    st.global.b32 [%out], %r3;",
            "bfi",
        ),
        _special(
            "shf",
            "    ld.global.b32 %r1, [%in];\n"
            "    ld.global.b32 %r2, [%in+4];\n"
            "    shf.l.wrap.b32 %r3, %r1, %r2, 5;\n"
            "    st.global.b32 [%out], %r3;",
            "shf",
        ),
        _special(
            "dp4a",
            "    ld.global.b32 %r1, [%in];\n"
            "    ld.global.b32 %r2, [%in+4];\n"
            "    ld.global.b32 %r3, [%in+8];\n"
            "    dp4a.s32.s32 %r4, %r1, %r2, %r3;\n"
            "    st.global.b32 [%out], %r4;",
            "dp4a",
        ),
        _special(
            "branch",
            "    ld.global.b32 %r1, [%in];\n"
            "    setp.gt.s32 %p1, %r1, 0;\n"
            "    @%p1 bra SKIP;\n"
            "    add.s32 %r1, %r1, 1;\n"
            "SKIP:\n"
            "    st.global.b32 [%out], %r1;",
            "bra",
        ),
        _special(
            "loop",
            # the trip count comes from the input, so unmasked this runs 2^31
            # times and costs more than the rest of the corpus together
            "    ld.global.b32 %r1, [%in];\n"
            "    and.b32 %r1, %r1, 63;\n"
            "    mov.u32 %r2, 0;\n"
            "LOOP:\n"
            "    add.s32 %r2, %r2, %r1;\n"
            "    sub.s32 %r1, %r1, 1;\n"
            "    setp.gt.s32 %p1, %r1, 0;\n"
            "    @%p1 bra LOOP;\n"
            "    st.global.b32 [%out], %r2;",
            "bra.loop",
        ),
        _special(
            "abs_compare",
            # the absolute value folds into the compare's operand, and `|R2|`
            # is a different encoding from `R2` that no other kernel reaches
            "    ld.global.f32 %f1, [%in];\n"
            "    ld.global.f32 %f2, [%in+4];\n"
            "    abs.f32 %f3, %f1;\n"
            "    setp.geu.f32 %p1, %f3, %f2;\n"
            "    selp.b32 %r7, 1, 0, %p1;\n"
            "    st.global.b32 [%out], %r7;",
            "setp.abs",
        ),
        _special(
            "abs_compare_second",
            "    ld.global.f32 %f1, [%in];\n"
            "    ld.global.f32 %f2, [%in+4];\n"
            "    abs.f32 %f3, %f2;\n"
            "    setp.geu.f32 %p1, %f1, %f3;\n"
            "    selp.b32 %r7, 1, 0, %p1;\n"
            "    st.global.b32 [%out], %r7;",
            "setp.abs2",
        ),
        _special(
            "neg_fma",
            # a negated source is likewise its own encoding of the same mnemonic
            "    ld.global.f32 %f1, [%in];\n"
            "    ld.global.f32 %f2, [%in+4];\n"
            "    neg.f32 %f3, %f1;\n"
            "    fma.rn.f32 %f4, %f3, %f2, %f2;\n"
            "    st.global.f32 [%out], %f4;",
            "fma.neg",
        ),
    ]

    return out
