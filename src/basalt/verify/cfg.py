# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Control flow, so a dependency can be followed past a branch.

Checking one straight-line block at a time is sound but blind: a value defined
before a loop and consumed inside it, or defined in one arm of a branch and
consumed after the join, is invisible. Real kernels are mostly those shapes, so
a checker that stops at branches misses most of what it exists to find.

The analysis here is a reaching-definitions dataflow over the whole kernel,
carrying two things along each edge:

*The minimum stall accumulated since the definition.* Different paths to the
same consumer accumulate different amounts, and only the smallest one matters:
if any path reaches the consumer too early, the program is wrong on that path.
Taking the minimum at every merge is what makes the result a statement about
every path rather than about a convenient one.

*Whether the scoreboard has been waited on, on every path.* The opposite
direction: a definition counts as satisfied only if something waits for it on
all incoming edges, because a single path that skips the wait is a hazard.

Loops terminate because the accumulated stall is monotone under merging and is
saturated at a bound well above the longest latency, so the lattice is finite.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..disasm import Instruction, Program, branch_target

__all__ = ["Block", "ControlFlowGraph", "ReachingDef", "build_cfg"]

# Once this much stall has accumulated, no latency on this architecture can
# still be outstanding, so the exact figure stops mattering. Saturating keeps
# the lattice finite and makes loops converge.
STALL_SATURATION = 512

_TERMINATORS = frozenset({"EXIT", "RET", "RTT", "BPT"})
_UNCONDITIONAL_BRANCH = frozenset({"BRA", "JMP", "BRX", "JMX"})
_CALLS = frozenset({"CALL", "CALL.ABS.NOINC", "RET"})
# Branches whose destination is computed rather than named. Their successors
# cannot be recovered from the listing, and pretending otherwise would make the
# analysis quietly unsound rather than visibly incomplete.
_INDIRECT = frozenset({"BRX", "JMX"})


@dataclass(frozen=True, slots=True)
class ReachingDef:
    """One definition that may reach a given point, and how it got there."""

    index: int  # instruction that produced it
    elapsed: int  # smallest stall accumulated since, over all paths
    satisfied: bool  # its scoreboard has been waited on, on every path
    barrier: int  # write_barrier it signalled
    # True when every path from the definition to here crosses an instruction
    # using the safe stall encoding, which waits for outstanding results as well
    # as elapsed cycles. Kept as a fact rather than folded into `elapsed`, since
    # a saturating counter cannot distinguish "waited" from "waited a long time".
    yielded: bool = False
    # True once this definition has been carried into another block. The gap to
    # a consumer then depends on which path was taken, so it is a minimum over
    # paths rather than a distance anything can be measured against.
    crossed: bool = False

    def merged_with(self, other: ReachingDef) -> ReachingDef:
        """Worst case of two paths reaching the same point."""
        return ReachingDef(
            index=self.index,
            elapsed=min(self.elapsed, other.elapsed),
            satisfied=self.satisfied and other.satisfied,
            barrier=self.barrier,
            yielded=self.yielded and other.yielded,
            crossed=self.crossed or other.crossed,
        )

    def advanced(self, stall: int, *, yielded: bool = False) -> ReachingDef:
        return replace(
            self,
            elapsed=min(STALL_SATURATION, self.elapsed + stall),
            yielded=self.yielded or yielded,
        )


@dataclass
class Block:
    """A straight-line run of instructions and where control can go next."""

    index: int
    start: int  # first instruction index, inclusive
    end: int  # one past the last, exclusive
    successors: list[int] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)
    # True when the block ends in a branch whose destination is not in the
    # listing, so its successors are unknown rather than empty
    indirect_exit: bool = False

    def __len__(self) -> int:
        return self.end - self.start

    def instructions(self, program: Program) -> list[Instruction]:
        return program.instructions[self.start : self.end]


@dataclass
class ControlFlowGraph:
    """Blocks and edges for one kernel."""

    program: Program
    blocks: list[Block]
    block_of: dict[int, int]  # instruction index -> block index

    @property
    def has_indirect_edges(self) -> bool:
        """True when some successor set could not be recovered.

        Callers should say so in their output. An analysis over a graph with
        missing edges is still sound for the paths it can see and silent about
        the ones it cannot, and the difference matters to anyone reading a
        clean result.
        """
        return any(b.indirect_exit for b in self.blocks)

    def entry(self) -> int:
        return 0

    def describe(self) -> str:
        edges = sum(len(b.successors) for b in self.blocks)
        note = ", some successors unknown" if self.has_indirect_edges else ""
        return f"{len(self.blocks)} blocks, {edges} edges{note}"


def _is_branch(instr: Instruction) -> bool:
    return instr.opcode.upper() in _UNCONDITIONAL_BRANCH


def _is_terminator(instr: Instruction) -> bool:
    return instr.opcode.upper() in _TERMINATORS


def build_cfg(program: Program) -> ControlFlowGraph:
    """Split a kernel into basic blocks and connect them."""
    n = len(program.instructions)
    if n == 0:
        return ControlFlowGraph(program=program, blocks=[], block_of={})

    # leaders: the entry, every branch target, and whatever follows a transfer
    leaders: set[int] = {0}
    for idx, instr in enumerate(program.instructions):
        label = branch_target(instr)
        if label is not None and (target := program.labels.get(label)) is not None:
            leaders.add(target)
        if (_is_branch(instr) or _is_terminator(instr)) and idx + 1 < n:
            leaders.add(idx + 1)
    for target in program.labels.values():
        if 0 <= target < n:
            leaders.add(target)

    starts = sorted(leaders)
    blocks: list[Block] = []
    block_of: dict[int, int] = {}
    for bi, start in enumerate(starts):
        end = starts[bi + 1] if bi + 1 < len(starts) else n
        blocks.append(Block(index=bi, start=start, end=end))
        for idx in range(start, end):
            block_of[idx] = bi

    index_of_start = {b.start: b.index for b in blocks}

    for block in blocks:
        last = program.instructions[block.end - 1]
        opcode = last.opcode.upper()

        if _is_terminator(last):
            continue

        if opcode in _INDIRECT:
            block.indirect_exit = True
            continue

        if _is_branch(last):
            label = branch_target(last)
            target = program.labels.get(label) if label else None
            if target is not None and target in index_of_start:
                block.successors.append(index_of_start[target])
            else:
                # a branch we cannot resolve is an unknown exit, not a dead end
                block.indirect_exit = True
            # a guarded branch may also fall through
            if last.predicate and block.end < len(program.instructions):
                block.successors.append(index_of_start[block.end])
            continue

        if block.end < len(program.instructions):
            block.successors.append(index_of_start[block.end])

    for block in blocks:
        for succ in block.successors:
            blocks[succ].predecessors.append(block.index)

    return ControlFlowGraph(program=program, blocks=blocks, block_of=block_of)
