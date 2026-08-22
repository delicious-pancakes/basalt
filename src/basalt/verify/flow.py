# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The dataflow state that travels along control-flow edges.

Two things propagate, and they merge in opposite directions because they answer
opposite questions.

*Reaching definitions* carry the smallest stall accumulated since the definition.
The question is "could a consumer be reached too early on some path", so merging
takes the minimum: a single fast path is enough to make the program wrong.

*Pending reads* carry instructions that signalled a read barrier and have not
been waited for. The question is "could an overwrite land while some path still
needs the old value", so merging takes the union: a single path that skipped the
wait is enough to make the program wrong.

Both directions are the conservative one for their question, which is what makes
a clean result mean something.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .cfg import ReachingDef
from .operands import RegRef

__all__ = ["FlowState", "PendingRead"]


@dataclass(frozen=True, slots=True)
class PendingRead:
    """An instruction that has not finished consuming its sources."""

    index: int
    registers: frozenset[RegRef]


@dataclass
class FlowState:
    """Everything that has to survive an edge between blocks."""

    # register -> definition index -> how that definition reaches here
    defs: dict[RegRef, dict[int, ReachingDef]] = field(default_factory=dict)
    # scoreboard -> reader index -> registers it is still reading
    reads: dict[int, dict[int, PendingRead]] = field(default_factory=dict)

    def copy(self) -> FlowState:
        return FlowState(
            defs={reg: dict(by_index) for reg, by_index in self.defs.items()},
            reads={sb: dict(by_index) for sb, by_index in self.reads.items()},
        )

    def crossing(self) -> FlowState:
        """This state as seen by a successor block.

        Every definition is marked as having crossed an edge. Past that point
        the distance to a consumer is a minimum over paths rather than a
        measurable gap, and a rule that needs a real distance has to know.
        """
        return FlowState(
            defs={
                reg: {i: replace(rd, crossed=True) for i, rd in by_index.items()}
                for reg, by_index in self.defs.items()
            },
            reads={sb: dict(by_index) for sb, by_index in self.reads.items()},
        )

    # ---- lattice -------------------------------------------------------

    def merge(self, other: FlowState) -> bool:
        """Merge `other` into this state. Returns True if anything changed."""
        changed = False

        for reg, incoming in other.defs.items():
            mine = self.defs.setdefault(reg, {})
            for index, rd in incoming.items():
                if (existing := mine.get(index)) is None:
                    mine[index] = rd
                    changed = True
                else:
                    merged = existing.merged_with(rd)
                    if merged != existing:
                        mine[index] = merged
                        changed = True

        for sb, incoming_reads in other.reads.items():
            mine_reads = self.reads.setdefault(sb, {})
            for index, pending in incoming_reads.items():
                if index not in mine_reads:
                    mine_reads[index] = pending
                    changed = True

        return changed

    # ---- transfer ------------------------------------------------------

    def satisfy(self, wait_mask: int) -> None:
        """Apply a wait: every definition outstanding on those scoreboards lands.

        Scoreboards are counters rather than flags, so one wait covers every
        signal outstanding on that scoreboard, for this instruction and for
        everything after it.
        """
        for reg, by_index in self.defs.items():
            for index, rd in list(by_index.items()):
                if not rd.satisfied and rd.barrier != 7 and (wait_mask >> rd.barrier) & 1:
                    by_index[index] = ReachingDef(
                        index=rd.index,
                        elapsed=rd.elapsed,
                        satisfied=True,
                        barrier=rd.barrier,
                    )
            self.defs[reg] = by_index

        for sb in list(self.reads):
            if (wait_mask >> sb) & 1:
                del self.reads[sb]

    def define(self, reg: RegRef, index: int, barrier: int) -> None:
        """Record a write, killing whatever previously defined that register."""
        self.defs[reg] = {
            index: ReachingDef(index=index, elapsed=0, satisfied=False, barrier=barrier)
        }

    def advance(self, stall: int, *, yielded: bool = False) -> None:
        """Charge one instruction's stall to every outstanding definition."""
        if stall == 0 and not yielded:
            return
        for reg, by_index in self.defs.items():
            self.defs[reg] = {i: rd.advanced(stall, yielded=yielded) for i, rd in by_index.items()}

    def begin_read(self, barrier: int, index: int, registers: frozenset[RegRef]) -> None:
        if barrier == 7 or not registers:
            return
        self.reads.setdefault(barrier, {})[index] = PendingRead(index, registers)

    def reaching(self, reg: RegRef) -> list[ReachingDef]:
        return sorted(self.defs.get(reg, {}).values(), key=lambda rd: rd.index)

    def pending_readers(self) -> list[tuple[int, PendingRead]]:
        return [
            (sb, pending) for sb, by_index in self.reads.items() for pending in by_index.values()
        ]
