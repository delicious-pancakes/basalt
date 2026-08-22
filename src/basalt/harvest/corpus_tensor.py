# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Corpus for the tensor-core and matrix-movement instructions.

Kept separate from the scalar corpus because these forms are the point. The
dense low-precision family (`QMMA` over the FP8, FP6 and FP4 types), the
block-scaled family that carries an explicit scale-factor operand (`QMMA.SF`,
`OMMA.SF`), and the sparse family (`IMMA.SP`) are where consumer Blackwell has
capability that is thinly documented, and reaching them is the demonstration
basalt is aiming at.

The operand-register counts below are fixed per shape and type. Where a count
is uncertain the form is emitted anyway: ptxas rejects it, the rejection is
recorded as a negative result, and a wrong guess costs one failed subprocess.
Guessing silently and getting an empty corpus is the outcome worth avoiding.
"""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import PTX_VERSION, Snippet

__all__ = ["generate_tensor"]


@dataclass(frozen=True, slots=True)
class MmaForm:
    """One `mma.sync` variant and the operand widths it expects."""

    shape: str  # e.g. m16n8k32
    atype: str
    btype: str
    ctype: str  # accumulator / result element type
    na: int  # number of 32-bit A fragment registers
    nb: int
    nc: int
    kind: str = ""  # e.g. kind::f8f6f4
    scale: str = ""  # e.g. ue8m0, implies block_scale
    scale_vec: str = ""  # e.g. scale_vec::4X
    layout: str = "row.col"


def _regs(prefix: str, count: int, base: int = 1) -> str:
    return "{" + ",".join(f"%{prefix}{base + i}" for i in range(count)) + "}"


def _mma_kernel(name: str, form: MmaForm) -> str:
    """Build a kernel around exactly one mma.sync.

    Fragments are loaded from global memory so nothing folds, and the first
    result register is stored so nothing is eliminated.
    """
    acc_reg = "f" if form.ctype.startswith("f") else "c"
    acc_ld = "f32" if form.ctype == "f32" else "b32"

    loads = []
    for i in range(form.na):
        loads.append(f"    ld.global.b32 %a{i + 1}, [%in+{4 * i}];")
    for i in range(form.nb):
        loads.append(f"    ld.global.b32 %b{i + 1}, [%in+{64 + 4 * i}];")
    for i in range(form.nc):
        loads.append(f"    ld.global.{acc_ld} %{acc_reg}{i + 1}, [%in+{128 + 4 * i}];")

    # scale-factor operands for the block-scaled family
    if form.scale:
        loads.append("    ld.global.b32 %s1, [%in+192];")
        loads.append("    ld.global.b32 %s2, [%in+196];")

    qualifiers = ".".join(x for x in (form.shape, form.layout, form.kind) if x)
    if form.scale:
        qualifiers += ".block_scale"
        if form.scale_vec:
            qualifiers += "." + form.scale_vec

    types = f"{form.ctype}.{form.atype}.{form.btype}.{form.ctype}"
    if form.scale:
        types += f".{form.scale}"

    dst = _regs(acc_reg, form.nc, base=form.nc + 1)
    operands = f"{dst},{_regs('a', form.na)},{_regs('b', form.nb)},{_regs(acc_reg, form.nc)}"
    if form.scale:
        operands += ",%s1,{0,0},%s2,{0,0}"

    store_type = "f32" if form.ctype == "f32" else "b32"
    body = "\n".join(
        [
            *loads,
            f"    mma.sync.aligned.{qualifiers}.{types} {operands};",
            f"    st.global.{store_type} [%out], %{acc_reg}{form.nc + 1};",
        ]
    )

    return f""".version {PTX_VERSION}
.target sm_120a
.address_size 64

.visible .entry {name}(.param .u64 pin, .param .u64 pout)
{{
    .reg .b32 %a<40>;
    .reg .b32 %b<40>;
    .reg .b32 %c<40>;
    .reg .b32 %s<8>;
    .reg .f32 %f<40>;
    .reg .b64 %in, %out;
    ld.param.u64  %in, [pin];
    cvta.to.global.u64 %in, %in;
    ld.param.u64  %out, [pout];
    cvta.to.global.u64 %out, %out;
{body}
    ret;
}}
"""


def _matrix_kernel(name: str, body: str) -> str:
    return f""".version {PTX_VERSION}
.target sm_120a
.address_size 64

.visible .entry {name}(.param .u64 pin, .param .u64 pout)
{{
    .reg .b32 %r<40>;
    .reg .b32 %m<8>;
    .reg .b64 %in, %out;
    .shared .align 16 .b8 tile[4096];
    ld.param.u64  %in, [pin];
    cvta.to.global.u64 %in, %in;
    ld.param.u64  %out, [pout];
    cvta.to.global.u64 %out, %out;
{body}
    ret;
}}
"""


# Dense forms. The FP8/FP6/FP4 types all route through kind::f8f6f4 on sm_120a
# and lower to QMMA; the f16/bf16/tf32 forms lower to HMMA and the integer
# forms to IMMA.
_LOW_PRECISION = ("e4m3", "e5m2", "e3m2", "e2m3", "e2m1")


def _dense_forms() -> list[MmaForm]:
    forms: list[MmaForm] = [
        MmaForm("m16n8k16", "f16", "f16", "f32", 4, 2, 4),
        MmaForm("m16n8k16", "f16", "f16", "f16", 4, 2, 2),
        MmaForm("m16n8k8", "f16", "f16", "f32", 2, 1, 4),
        MmaForm("m16n8k16", "bf16", "bf16", "f32", 4, 2, 4),
        MmaForm("m16n8k8", "bf16", "bf16", "f32", 2, 1, 4),
        MmaForm("m16n8k8", "tf32", "tf32", "f32", 4, 2, 4),
        MmaForm("m16n8k4", "tf32", "tf32", "f32", 2, 1, 4),
        MmaForm("m16n8k32", "s8", "s8", "s32", 4, 2, 4),
        MmaForm("m16n8k32", "u8", "u8", "s32", 4, 2, 4),
        MmaForm("m16n8k32", "s8", "u8", "s32", 4, 2, 4),
        MmaForm("m16n8k16", "s8", "s8", "s32", 2, 1, 4),
        MmaForm("m8n8k16", "s8", "s8", "s32", 1, 1, 2),
        MmaForm("m16n8k64", "s4", "s4", "s32", 4, 2, 4),
        MmaForm("m16n8k32", "s4", "s4", "s32", 2, 1, 4),
        MmaForm("m16n8k256", "b1", "b1", "s32", 4, 2, 4),
    ]
    # every ordered pair of the low-precision types, mixed inputs included:
    # ptxas accepts asymmetric operand types here and each pair is a distinct
    # QMMA type-code combination worth having in the database
    for a in _LOW_PRECISION:
        for b in _LOW_PRECISION:
            forms.append(MmaForm("m16n8k32", a, b, "f32", 4, 2, 4, kind="kind::f8f6f4"))
    return forms


def _block_scaled_forms() -> list[MmaForm]:
    """The scale-factor family: an extra operand carrying a per-block exponent."""
    forms: list[MmaForm] = []
    for a in _LOW_PRECISION:
        for scale in ("ue8m0", "ue4m3"):
            forms.append(
                MmaForm("m16n8k32", a, a, "f32", 4, 2, 4, kind="kind::mxf8f6f4", scale=scale)
            )
    for scale, vec in (("ue8m0", ""), ("ue4m3", "scale_vec::4X"), ("ue8m0", "scale_vec::2X")):
        forms.append(
            MmaForm(
                "m16n8k64",
                "e2m1",
                "e2m1",
                "f32",
                4,
                2,
                4,
                kind="kind::mxf4nvf4" if vec else "kind::mxf4",
                scale=scale,
                scale_vec=vec,
            )
        )
    return forms


def generate_tensor() -> list[Snippet]:
    out: list[Snippet] = []

    for form in _dense_forms() + _block_scaled_forms():
        tag = f"{form.shape}_{form.atype}_{form.btype}_{form.ctype}"
        if form.kind:
            tag += "_" + form.kind.replace("kind::", "")
        if form.scale:
            tag += "_" + form.scale
        if form.scale_vec:
            tag += "_" + form.scale_vec.replace("scale_vec::", "v")
        name = f"k_mma_{tag}"
        family = "mma.block_scaled" if form.scale else "mma.dense"
        out.append(
            Snippet(
                name,
                _mma_kernel(name, form),
                family,
                "mma.sync",
                f"{form.atype}.{form.btype}.{form.ctype}",
                (form.shape, form.kind, form.scale, form.scale_vec),
            )
        )

    # sparse variants carry a metadata operand and a selector
    for shape, atype, ctype, na, nb, nc in (
        ("m16n8k32", "f16", "f32", 2, 2, 4),
        ("m16n8k64", "s8", "s32", 4, 4, 4),
        ("m16n8k64", "e4m3", "f32", 4, 4, 4),
        ("m16n8k128", "e2m1", "f32", 4, 4, 4),
    ):
        acc = "f" if ctype == "f32" else "c"
        acc_ld = "f32" if ctype == "f32" else "b32"
        loads = [f"    ld.global.b32 %a{i + 1}, [%in+{4 * i}];" for i in range(na)]
        loads += [f"    ld.global.b32 %b{i + 1}, [%in+{64 + 4 * i}];" for i in range(nb)]
        loads += [f"    ld.global.{acc_ld} %{acc}{i + 1}, [%in+{128 + 4 * i}];" for i in range(nc)]
        loads.append("    ld.global.b32 %s1, [%in+192];")
        kind = ".kind::f8f6f4" if atype in _LOW_PRECISION else ""
        body = "\n".join(
            [
                *loads,
                f"    mma.sp.sync.aligned.{shape}.row.col{kind}.{ctype}.{atype}.{atype}.{ctype} "
                f"{_regs(acc, nc, base=nc + 1)},{_regs('a', na)},{_regs('b', nb)},"
                f"{_regs(acc, nc)},%s1,0x0;",
                f"    st.global.{acc_ld} [%out], %{acc}{nc + 1};",
            ]
        )
        name = f"k_mmasp_{shape}_{atype}_{ctype}"
        out.append(
            Snippet(name, _mma_kernel_raw(name, body), "mma.sparse", "mma.sp.sync", atype, (shape,))
        )

    # matrix load / store / transpose, which feed the fragments above
    for count in ("x1", "x2", "x4"):
        n = {"x1": 1, "x2": 2, "x4": 4}[count]
        for trans in ("", ".trans"):
            body = (
                "    mov.u32 %m1, tile;\n"
                f"    ldmatrix.sync.aligned.m8n8.{count}{trans}.shared.b16 "
                f"{_regs('r', n)}, [%m1];\n"
                "    st.global.b32 [%out], %r1;"
            )
            name = f"k_ldsm_{count}{trans.replace('.', '_')}"
            out.append(
                Snippet(
                    name,
                    _matrix_kernel(name, body),
                    "matrix",
                    "ldmatrix",
                    "b16",
                    (count, trans.strip(".")),
                )
            )

            load_text = "\n".join(f"    ld.global.b32 %r{i + 1}, [%in+{4 * i}];" for i in range(n))
            regs = _regs("r", n)
            shape = f"m8n8.{count}{trans}"
            body = (
                f"{load_text}\n"
                "    mov.u32 %m1, tile;\n"
                f"    stmatrix.sync.aligned.{shape}.shared.b16 [%m1], {regs};\n"
                "    st.global.b32 [%out], %r1;"
            )
            name = f"k_stsm_{count}{trans.replace('.', '_')}"
            out.append(
                Snippet(
                    name,
                    _matrix_kernel(name, body),
                    "matrix",
                    "stmatrix",
                    "b16",
                    (count, trans.strip(".")),
                )
            )

    # m16n8 ldmatrix over 8-bit elements, and the register-to-register transpose
    for count in ("x1", "x2", "x4"):
        n = {"x1": 1, "x2": 2, "x4": 4}[count]
        body = (
            "    mov.u32 %m1, tile;\n"
            f"    ldmatrix.sync.aligned.m16n16.{count}.trans.shared.b8 {_regs('r', n)}, [%m1];\n"
            "    st.global.b32 [%out], %r1;"
        )
        name = f"k_ldsm_m16n16_{count}_b8"
        out.append(Snippet(name, _matrix_kernel(name, body), "matrix", "ldmatrix", "b8", (count,)))

    body = (
        "    ld.global.b32 %r1, [%in];\n"
        "    movmatrix.sync.aligned.m8n8.trans.b16 %r2, %r1;\n"
        "    st.global.b32 [%out], %r2;"
    )
    out.append(
        Snippet("k_movmatrix", _matrix_kernel("k_movmatrix", body), "matrix", "movmatrix", "b16")
    )

    return out


def _mma_kernel_raw(name: str, body: str) -> str:
    """Same preamble as _mma_kernel but with a caller-supplied body."""
    return f""".version {PTX_VERSION}
.target sm_120a
.address_size 64

.visible .entry {name}(.param .u64 pin, .param .u64 pout)
{{
    .reg .b32 %a<40>;
    .reg .b32 %b<40>;
    .reg .b32 %c<40>;
    .reg .b32 %s<8>;
    .reg .f32 %f<40>;
    .reg .b64 %in, %out;
    ld.param.u64  %in, [pin];
    cvta.to.global.u64 %in, %in;
    ld.param.u64  %out, [pout];
    cvta.to.global.u64 %out, %out;
{body}
    ret;
}}
"""
