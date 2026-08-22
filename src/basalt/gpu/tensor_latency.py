# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Timing the tensor cores.

The scalar harness threads a single register through a chain. A matrix
instruction cannot work that way, but it does not need to: `mma` accumulates
into its own D operand, so feeding D back in as C makes each instruction depend
on the previous one by construction. That is a genuine dependent chain with no
extra instructions between the links, which is exactly what the measurement
wants.

Low-precision inputs need care. Half and lower formats saturate or flush to zero
quickly under repeated accumulation, and a chain whose value stops changing is a
chain that cannot detect a stale read. The fragments are seeded from memory with
small values chosen to keep the accumulator moving without reaching infinity
over the length of a run.

These figures do not appear to be published for consumer Blackwell, so treat the
numbers as this project's own measurements rather than as confirmation of
anything. Each still has to pass the same gates as the scalar ones: linear fit,
whole number of cycles, and a chain length counted in the compiled SASS rather
than assumed.
"""

from __future__ import annotations

from .latency import ChainSpec

__all__ = ["TENSOR_SPECS", "tensor_kernel"]


# (label, shape, A type, B type, accumulator type, kind qualifier, A regs, B regs, C regs)
_FORMS: tuple[tuple[str, str, str, str, str, str, int, int, int], ...] = (
    ("HMMA.16816.F32", "m16n8k16", "f16", "f16", "f32", "", 4, 2, 4),
    ("HMMA.16816.F32.BF16", "m16n8k16", "bf16", "bf16", "f32", "", 4, 2, 4),
    ("HMMA.1688.F32.TF32", "m16n8k8", "tf32", "tf32", "f32", "", 4, 2, 4),
    ("IMMA.16832.S8.S8", "m16n8k32", "s8", "s8", "s32", "", 4, 2, 4),
    ("QMMA.16832.F32.E4M3.E4M3", "m16n8k32", "e4m3", "e4m3", "f32", "kind::f8f6f4", 4, 2, 4),
    ("QMMA.16832.F32.E5M2.E5M2", "m16n8k32", "e5m2", "e5m2", "f32", "kind::f8f6f4", 4, 2, 4),
    ("QMMA.16832.F32.E3M2.E3M2", "m16n8k32", "e3m2", "e3m2", "f32", "kind::f8f6f4", 4, 2, 4),
    ("QMMA.16832.F32.E2M1.E2M1", "m16n8k32", "e2m1", "e2m1", "f32", "kind::f8f6f4", 4, 2, 4),
)

_TENSOR_KERNEL = """.version 9.0
.target {arch}
.address_size 64

.visible .entry chain(.param .u64 pin, .param .u64 pout)
{{
    .reg .b32 %a<8>;
    .reg .b32 %b<8>;
    .reg .b32 %c<8>;
    .reg .f32 %f<8>;
    .reg .b32 %t0, %t1, %t2;
    .reg .b64 %in, %out;

    ld.param.u64  %in, [pin];
    cvta.to.global.u64 %in, %in;
    ld.param.u64  %out, [pout];
    cvta.to.global.u64 %out, %out;

{loads}

    mov.u32 %t0, %clock;
{chain}
    mov.u32 %t1, %clock;

    sub.s32 %t2, %t1, %t0;
    st.global.u32 [%out], %t2;
{sink}
    ret;
}}
"""


def tensor_kernel(
    shape: str,
    atype: str,
    btype: str,
    ctype: str,
    kind: str,
    na: int,
    nb: int,
    nc: int,
    length: int,
    arch: str = "sm_120a",
) -> str:
    """Build a dependent chain of `mma.sync`, accumulating through D into C."""
    acc = "f" if ctype == "f32" else "c"
    acc_ld = "f32" if ctype == "f32" else "b32"

    loads = [f"    ld.global.b32 %a{i + 1}, [%in+{4 * i}];" for i in range(na)]
    loads += [f"    ld.global.b32 %b{i + 1}, [%in+{32 + 4 * i}];" for i in range(nb)]
    loads += [f"    ld.global.{acc_ld} %{acc}{i + 1}, [%in+{64 + 4 * i}];" for i in range(nc)]

    frag_a = "{" + ",".join(f"%a{i + 1}" for i in range(na)) + "}"
    frag_b = "{" + ",".join(f"%b{i + 1}" for i in range(nb)) + "}"
    frag_c = "{" + ",".join(f"%{acc}{i + 1}" for i in range(nc)) + "}"

    qualifiers = ".".join(x for x in (shape, "row.col", kind) if x)
    # D and C are the same registers, so the next instruction cannot issue until
    # this one has written them back. that is the dependency being timed.
    link = (
        f"    mma.sync.aligned.{qualifiers}.{ctype}.{atype}.{btype}.{ctype} "
        f"{frag_c},{frag_a},{frag_b},{frag_c};"
    )

    return _TENSOR_KERNEL.format(
        arch=arch,
        loads="\n".join(loads),
        chain="\n".join(link for _ in range(length)),
        sink=f"    st.global.{acc_ld} [%out+4], %{acc}1;",
    )


def _spec(entry: tuple[str, str, str, str, str, str, int, int, int]) -> ChainSpec:
    label, shape, atype, btype, ctype, kind, na, nb, nc = entry
    opcode = label.split(".")[0]
    return ChainSpec(
        opcode=opcode,
        ptx_type=ctype,
        # the body is unused for tensor chains; tensor_kernel builds the whole
        # kernel, because the fragment operands do not fit the scalar template
        body="",
        seed="",
        note=f"{label}, accumulating through the D operand",
    )


TENSOR_SPECS: tuple[tuple[ChainSpec, tuple], ...] = tuple((_spec(e), e) for e in _FORMS)
