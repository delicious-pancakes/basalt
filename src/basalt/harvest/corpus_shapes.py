# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Kernels with real control flow, for the round trip rather than the harvest.

The generated corpus is deliberately narrow: one or two instructions of body per
kernel, so that a form appears in isolation and the prober can attribute a bit to
it. That is right for building an instruction database and wrong for exercising a
scheduler. Almost nothing in it has a loop, a barrier, a nested branch, or shared
memory that is actually addressable, and those are exactly the shapes where the
control bits get interesting: a dependency carried around a back edge, a value
live across a branch, a store that has to land before a `bar.sync`.

So these are separate, hand-written, and shaped like code somebody would run.
Each keeps the same `(pin, pout)` signature as the corpus so the round trip needs
no special case, and each is written to be genuinely runnable rather than merely
compilable, which the shared-memory forms in the main corpus are not.

They are harvested as well as scheduled. Written for the round trip, they turn
out to be the only kernels that reach a predicated form or a branch displacement,
and a database that has never seen one has to refuse it.
"""

from __future__ import annotations

from .corpus import Snippet

__all__ = ["generate_shapes"]

_PREAMBLE = """.version 9.0
.target sm_120a
.address_size 64
"""


def _kernel(name: str, body: str, *, shared: str = "", extra_regs: str = "") -> Snippet:
    source = f"""{_PREAMBLE}{shared}
.visible .entry {name}(.param .u64 pin, .param .u64 pout)
{{
    .reg .pred  %p<8>;
    .reg .b16   %h<8>;
    .reg .b32   %r<24>;
    .reg .b64   %d<16>;
    .reg .f32   %f<16>;
    .reg .f64   %g<8>;
    .reg .b64   %in, %out;
{extra_regs}
    ld.param.u64  %in, [pin];
    cvta.to.global.u64 %in, %in;
    ld.param.u64  %out, [pout];
    cvta.to.global.u64 %out, %out;
{body}
    ret;
}}
"""
    return Snippet(name=name, ptx=source, family="shape", op=name, ptx_type="mixed")


def generate_shapes() -> list[Snippet]:
    """Kernels whose control flow is the point, not their opcodes."""
    return [
        # A counted loop with an accumulator carried around the back edge. The
        # shape that first exposed the guard-predicate requirement.
        _kernel(
            "s_loop_accumulate",
            """    ld.global.u32 %r1, [%in];
    and.b32 %r1, %r1, 31;
    add.s32 %r1, %r1, 1;
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
LOOP:
    mad.lo.s32 %r2, %r2, 3, %r3;
    add.s32 %r3, %r3, 1;
    setp.lt.s32 %p1, %r3, %r1;
    @%p1 bra LOOP;
    st.global.u32 [%out], %r2;""",
        ),
        # Two accumulators updated in the same loop, so the back edge carries
        # more than one live value and they interleave.
        _kernel(
            "s_loop_two_accumulators",
            """    ld.global.u32 %r1, [%in];
    and.b32 %r1, %r1, 15;
    add.s32 %r1, %r1, 2;
    mov.u32 %r2, 1;
    mov.u32 %r4, 0;
    mov.u32 %r3, 0;
LOOP2:
    mad.lo.s32 %r2, %r2, 5, %r3;
    add.s32 %r4, %r4, %r2;
    add.s32 %r3, %r3, 1;
    setp.lt.s32 %p1, %r3, %r1;
    @%p1 bra LOOP2;
    add.s32 %r5, %r2, %r4;
    st.global.u32 [%out], %r5;""",
        ),
        # A branch whose two sides define the same register, so both definitions
        # reach the join and both have to be covered.
        _kernel(
            "s_branch_join",
            """    ld.global.u32 %r1, [%in];
    ld.global.u32 %r2, [%in+4];
    setp.gt.u32 %p1, %r1, %r2;
    @%p1 bra BIG;
    mul.lo.s32 %r3, %r1, %r2;
    bra JOIN;
BIG:
    add.s32 %r3, %r1, %r2;
JOIN:
    mad.lo.s32 %r4, %r3, %r3, %r1;
    st.global.u32 [%out], %r4;""",
        ),
        # A branch inside a loop, so the back edge and the join interact.
        _kernel(
            "s_loop_with_branch",
            """    ld.global.u32 %r1, [%in];
    and.b32 %r1, %r1, 15;
    add.s32 %r1, %r1, 2;
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
LOOP3:
    and.b32 %r4, %r3, 1;
    setp.eq.s32 %p1, %r4, 0;
    @%p1 bra EVEN;
    mul.lo.s32 %r2, %r2, 3;
    bra NEXT;
EVEN:
    add.s32 %r2, %r2, %r3;
NEXT:
    add.s32 %r3, %r3, 1;
    setp.lt.s32 %p2, %r3, %r1;
    @%p2 bra LOOP3;
    st.global.u32 [%out], %r2;""",
        ),
        # Shared memory that is actually addressable, written then read back
        # across a barrier. The corpus reaches `LDS` only through an address in
        # the wrong space, which compiles and cannot run.
        _kernel(
            "s_shared_roundtrip",
            """    ld.global.u32 %r1, [%in];
    mov.u32 %r5, tile;
    st.shared.u32 [%r5], %r1;
    bar.sync 0;
    ld.shared.u32 %r2, [%r5];
    mul.lo.s32 %r3, %r2, 3;
    st.global.u32 [%out], %r3;""",
            shared="\n.shared .align 4 .b8 tile[256];\n",
        ),
        # A barrier with real traffic on both sides of it.
        _kernel(
            "s_shared_stencil",
            """    ld.global.u32 %r1, [%in];
    ld.global.u32 %r2, [%in+4];
    mov.u32 %r5, tile2;
    st.shared.u32 [%r5], %r1;
    st.shared.u32 [%r5+4], %r2;
    bar.sync 0;
    ld.shared.u32 %r6, [%r5];
    ld.shared.u32 %r7, [%r5+4];
    add.s32 %r8, %r6, %r7;
    mul.lo.s32 %r9, %r8, %r6;
    st.global.u32 [%out], %r9;""",
            shared="\n.shared .align 4 .b8 tile2[256];\n",
        ),
        # Floating point around a loop, which schedules differently from integer
        # because the pipe is narrower.
        _kernel(
            "s_loop_float",
            """    ld.global.f32 %f1, [%in];
    ld.global.f32 %f2, [%in+4];
    mov.f32 %f3, 0f3F800000;
    mov.u32 %r3, 0;
LOOP4:
    fma.rn.f32 %f3, %f3, %f2, %f1;
    add.s32 %r3, %r3, 1;
    setp.lt.s32 %p1, %r3, 8;
    @%p1 bra LOOP4;
    st.global.f32 [%out], %f3;""",
        ),
        # Double precision around a loop. fp64 is scoreboarded and slow, so the
        # back edge and the scoreboard have to agree.
        _kernel(
            "s_loop_double",
            """    ld.global.f64 %g1, [%in];
    ld.global.f64 %g2, [%in+8];
    mov.f64 %g3, 0d3FF0000000000000;
    mov.u32 %r3, 0;
LOOP5:
    fma.rn.f64 %g3, %g3, %g2, %g1;
    add.s32 %r3, %r3, 1;
    setp.lt.s32 %p1, %r3, 4;
    @%p1 bra LOOP5;
    st.global.f64 [%out], %g3;""",
        ),
        # A long dependent chain with no branches at all, which is where the
        # stall placement has the least room to spill into.
        _kernel(
            "s_long_chain",
            """    ld.global.u32 %r1, [%in];
    mad.lo.s32 %r2, %r1, %r1, %r1;
    mad.lo.s32 %r3, %r2, %r2, %r1;
    mad.lo.s32 %r4, %r3, %r3, %r2;
    mad.lo.s32 %r5, %r4, %r4, %r3;
    mad.lo.s32 %r6, %r5, %r5, %r4;
    mad.lo.s32 %r7, %r6, %r6, %r5;
    mad.lo.s32 %r8, %r7, %r7, %r6;
    st.global.u32 [%out], %r8;""",
        ),
        # Predicated writes to a register a later instruction reads, so the
        # earlier definition survives on one path.
        _kernel(
            "s_predicated_writes",
            """    ld.global.u32 %r1, [%in];
    ld.global.u32 %r2, [%in+4];
    mul.lo.s32 %r3, %r1, %r2;
    setp.gt.u32 %p1, %r1, %r2;
    @%p1 mul.lo.s32 %r3, %r3, 7;
    setp.lt.u32 %p2, %r1, %r2;
    @%p2 add.s32 %r3, %r3, 11;
    mad.lo.s32 %r4, %r3, %r1, %r2;
    st.global.u32 [%out], %r4;""",
        ),
        # A warp shuffle feeding arithmetic, the shape the warp-aggregated
        # atomics use.
        _kernel(
            "s_shuffle_chain",
            """    ld.global.u32 %r1, [%in];
    mov.u32 %r6, 31;
    mov.u32 %r7, -1;
    shfl.sync.idx.b32 %r2, %r1, %r6, 31, %r7;
    mul.lo.s32 %r3, %r2, 5;
    shfl.sync.idx.b32 %r4, %r3, %r6, 31, %r7;
    add.s32 %r5, %r4, %r2;
    st.global.u32 [%out], %r5;""",
        ),
        # `BMSK` is in the database from the -O0 harvest, where `bfi.b32`
        # lowers to it, and at -O3 that kernel folds it away. Written directly
        # it survives, which puts the opcode inside the hardware round trip
        # instead of only inside the database.
        _kernel(
            "s_bit_mask",
            """    ld.global.u32 %r1, [%in];
    ld.global.u32 %r2, [%in+4];
    bmsk.clamp.b32 %r3, %r1, %r2;
    bmsk.wrap.b32 %r4, %r1, %r2;
    add.s32 %r5, %r3, %r4;
    st.global.u32 [%out], %r5;""",
        ),
        # Nested loops, so one back edge sits inside another.
        _kernel(
            "s_nested_loops",
            """    ld.global.u32 %r1, [%in];
    and.b32 %r1, %r1, 7;
    add.s32 %r1, %r1, 2;
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
OUTER:
    mov.u32 %r4, 0;
INNER:
    mad.lo.s32 %r2, %r2, 3, %r4;
    add.s32 %r4, %r4, 1;
    setp.lt.s32 %p1, %r4, %r1;
    @%p1 bra INNER;
    add.s32 %r3, %r3, 1;
    setp.lt.s32 %p2, %r3, %r1;
    @%p2 bra OUTER;
    st.global.u32 [%out], %r2;""",
        ),
    ]
