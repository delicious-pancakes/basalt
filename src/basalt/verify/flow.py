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
        if not wait_mask:
            return
        for by_index in self.defs.values():
            for index, rd in by_index.items():
                if not rd.satisfied and rd.barrier != 7 and (wait_mask >> rd.barrier) & 1:
                    # every field carried across: rebuilding without `yielded`
                    # and `crossed` reset both on every wait, which made a
                    # definition that had crossed an edge look like one that
                    # had not, and the gap a distance it is not
                    by_index[index] = ReachingDef(
                        rd.index,
                        rd.elapsed,
                        True,
                        rd.barrier,
                        rd.yielded,
                        rd.crossed,
                    )

        for sb in list(self.reads):
            if (wait_mask >> sb) & 1:
                del self.reads[sb]

    def define(self, reg: RegRef, index: int, barrier: int, *, conditional: bool = False) -> None:
        """Record a write.

        An unconditional write kills whatever previously defined the register.
        A predicated one does not: `@!P0 FMUL R7, R7, c` leaves R7 holding
        whatever produced it when the guard is false, so both that instruction
        and the earlier producer reach any later reader and both have to be
        covered. Treating a predicated write as a kill is how basalt dropped the
        wait on a `MUFU.SQRT` whose result a store read directly whenever the
        guard came out false, which the GPU reported as a non-deterministic
        square root and no static check would have shown.
        """
        fresh = ReachingDef(index=index, elapsed=0, satisfied=False, barrier=barrier)
        if conditional:
            self.defs.setdefault(reg, {})[index] = fresh
        else:
            self.defs[reg] = {index: fresh}

    def adopt(self, barrier: int, is_same_unit) -> None:
        """Let a new scoreboard cover earlier unscoreboarded results from its unit.

        A variable-latency unit returns results in the order it was given work,
        so a wait on something issued later covers everything the same unit
        still owes. `ptxas` schedules two consecutive `SHFL.IDX` and puts a
        scoreboard on the second only, then waits on that one before reading
        either result.

        Without this the checker reports the first shuffle as a result nothing
        can wait for, which is a false positive against the vendor's own output
        and therefore the kind of finding that makes every other finding
        worthless.
        """
        for by_index in self.defs.values():
            for i, rd in by_index.items():
                if rd.barrier == 7 and is_same_unit(rd.index):
                    by_index[i] = replace(rd, barrier=barrier)

    def advance(self, stall: int, *, yielded: bool = False) -> None:
        """Charge one instruction's stall to every outstanding definition."""
        if stall == 0 and not yielded:
            return
        for by_index in self.defs.values():
            for i, rd in by_index.items():
                moved = rd.advanced(stall, yielded=yielded)
                if moved is not rd:
                    by_index[i] = moved

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
