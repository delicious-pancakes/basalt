# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Assigning the control bits, rather than only checking them.

The verifier answers "is this schedule safe". This answers "what would a safe
schedule be", from the same latency model and the same dependence rules. That
shared basis is the point: a scheduler and a checker built on different
assumptions are two bugs waiting to find each other, so anything this produces
is handed straight back to the verifier, and then to the hardware.

Instruction order is left exactly as it was; only the control word is rewritten.
That is the design rather than a shortcut. Reordering would make it impossible to
say whether a change in behaviour came from the control bits or from the new
order, and the control bits are the thing under test.

The assignment runs to a fixed point. Each pass walks the block, works out how
much every consumer is short by, and adds that much stall to the instructions
immediately before it. Adding stall never reduces any other gap, and every stall
field is bounded, so the loop can only run until nothing is short.

Where a requirement is larger than one stall field can express, the shortfall is
pushed back onto earlier instructions, which is what the vendor compiler does
when it pads with maximum-stall NOPs. If it cannot be placed at all, that is
reported rather than silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..disasm import Instruction, Program
from ..encoding import NO_BARRIER, STALL_YIELD, Word
from ..verify.cfg import build_cfg
from ..verify.latency import GUARD_CYCLES, LatencyClass, LatencyModel
from ..verify.observed import ObservedStalls
from ..verify.operands import RegRef, operand_access

__all__ = [
    "SCOREBOARDS",
    "STALL_MAX",
    "YIELD_COST",
    "ScheduleResult",
    "issue_cycles",
    "schedule_program",
]

# a zero stall is a long wait, not no wait: ~37 cycles (finding 1)
YIELD_COST = 37

# Four bits of stall, so 15 is the most a single instruction can request.
STALL_MAX = 15

# Six scoreboards, indices 0..5; 7 means "none" and 6 is left alone.
SCOREBOARDS = 6

# A gap this large is covered by any latency on this architecture, so once an
# instruction is this far behind there is nothing left to satisfy.
SATURATION = 512

# Termination guard. The lattice is finite, so reaching this means a bug.
MAX_PASSES = 64

# transfers to code this analysis has not read, so nothing may be in flight
_OPAQUE_TRANSFERS = frozenset({"CALL", "RET", "BRX", "JMX", "RTT", "BPT"})

# instructions ptxas never gives the zero-stall encoding, and its floor for each.
# basalt reaches for that encoding as a fallback, so these need somewhere else
_NEVER_ZERO_STALL: dict[str, int] = {"EXIT": 5, "RET": 5, "CALL": 5, "BAR": 6}


@dataclass
class ScheduleResult:
    """A rescheduled program, and what could not be arranged."""

    words: list[Word]
    passes: int = 0
    stalls_added: int = 0
    scoreboards_used: int = 0
    # shared an already-busy scoreboard: over-synchronised, so counted not hidden
    scoreboards_shared: int = 0
    unplaceable: list[str] = field(default_factory=list)
    out_of_scoreboards: list[str] = field(default_factory=list)
    # Instructions given the safe stall encoding because their dependency could
    # not be shown covered any other way. Correct but slow, so it is counted.
    yielded: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.out_of_scoreboards

    def summary(self) -> str:
        verdict = "complete" if self.ok else f"{len(self.out_of_scoreboards)} unallocatable"
        return (
            f"{len(self.words)} instructions, {self.passes} passes, "
            f"{self.stalls_added} stall cycles, {self.scoreboards_used} scoreboards, "
            f"{self.scoreboards_shared} shared, {len(self.yielded)} safe stalls: {verdict}"
        )


@dataclass(slots=True)
class _Producer:
    index: int
    opcode: str
    mnemonic: str
    kind: LatencyClass
    barrier: int = NO_BARRIER


def _requirement(
    producer: str,
    consumer: str,
    cycles: int,
    observed: ObservedStalls | None,
    *,
    guard: bool = False,
) -> int:
    """Cycles this pairing needs, preferring evidence about the exact pairing.

    A guard is its own pairing. It gates issue rather than feeding an operand
    port, so it needs about two and a half times the lead of the same predicate
    read as data, and getting this wrong produces a schedule that runs and
    quietly returns the wrong answer.
    """
    key = ("@" if guard else "") + consumer
    if observed is not None:
        evidence = observed.requirement(producer, key)
        if evidence is not None:
            return evidence.minimum
    return GUARD_CYCLES if guard else cycles


def _scoreboarded_requirement(
    producer_mnemonic: str,
    consumer: str,
    observed: ObservedStalls | None,
    *,
    guard: bool = False,
) -> int:
    """Cycles a waited-on scoreboarded producer still needs before its consumer.

    Zero when nothing was observed, because there is no basis for inventing one
    and the wait already covers the bulk of it.
    """
    if observed is None:
        return 0
    evidence = observed.scoreboarded_minimum(producer_mnemonic, ("@" if guard else "") + consumer)
    return evidence.minimum if evidence is not None else 0


def _live_out(cfg, program) -> list[set[RegRef]]:
    """Registers each block defines that something after it may read.

    The ordinary backwards liveness fixed point, restricted to what this needs:
    a block's live-out set is the union of its successors' live-in sets, and a
    live-in is anything read before being written. A definition in the set is
    one whose consumer is in another block, which is exactly the case the
    per-block stall analysis cannot see.
    """
    used: list[set[RegRef]] = []
    written: list[set[RegRef]] = []
    for block in cfg.blocks:
        reads: set[RegRef] = set()
        writes: set[RegRef] = set()
        for index in range(block.start, block.end):
            instruction = program.instructions[index]
            if instruction.word is None:
                continue
            access = operand_access(instruction.mnemonic, instruction.operands)
            reads |= access.real_uses - writes
            writes |= access.real_defs
        used.append(reads)
        written.append(writes)

    live_in: list[set[RegRef]] = [set() for _ in cfg.blocks]
    live_out: list[set[RegRef]] = [set() for _ in cfg.blocks]
    for _ in range(len(cfg.blocks) + 1):
        changed = False
        for block in reversed(cfg.blocks):
            outgoing: set[RegRef] = set()
            for successor in block.successors:
                outgoing |= live_in[successor]
            incoming = used[block.index] | (outgoing - written[block.index])
            if outgoing != live_out[block.index] or incoming != live_in[block.index]:
                live_out[block.index] = outgoing
                live_in[block.index] = incoming
                changed = True
        if not changed:
            break

    # a block with unknown successors could reach anything, so nothing about it
    # can be assumed to be dead
    for block in cfg.blocks:
        if getattr(block, "successors_unknown", False):
            live_out[block.index] = live_out[block.index] | written[block.index]
    return live_out


def _short_pair(block, program, stalls, pinned, model, observed) -> bool:
    """Is any dependency inside this block short of what it needs?

    The same walk the assignment fixed point does, asking only whether anything
    is short rather than fixing it. Sharing the requirement function is the
    point: a shrink pass judged by a looser rule than the one that placed the
    cycles would quietly undo the placement.
    """
    last_def: dict[RegRef, list[_Producer]] = {}
    elapsed: dict[int, int] = {}

    for index in range(block.start, block.end):
        instr = program.instructions[index]
        if instr.word is None:
            continue
        access = operand_access(instr.mnemonic, instr.operands)

        for reg in access.real_uses:
            for producer in last_def.get(reg, ()):
                record = model.lookup(producer.opcode)
                if producer.kind is LatencyClass.FIXED:
                    needed = _requirement(
                        producer.mnemonic,
                        instr.opcode,
                        record.cycles,
                        observed,
                        guard=reg == access.guard,
                    )
                else:
                    needed = _scoreboarded_requirement(
                        producer.mnemonic, instr.opcode, observed, guard=reg == access.guard
                    )
                if needed and elapsed.get(producer.index, 0) < needed:
                    return True

        charge = SATURATION if index in pinned else stalls[index]
        for key in elapsed:
            elapsed[key] = min(SATURATION, elapsed[key] + charge)

        produced = _Producer(
            index,
            instr.opcode,
            instr.mnemonic,
            model.lookup(instr.opcode).kind,
        )
        for reg in access.real_defs:
            if access.guard:
                last_def.setdefault(reg, []).append(produced)
            else:
                last_def[reg] = [produced]
            elapsed[index] = stalls[index]
    return False


def _place_stall(
    stalls: list[int],
    pinned: set[int],
    producer: int,
    consumer: int,
    amount: int,
) -> int:
    """Add `amount` cycles of stall between a producer and its consumer.

    The window is `[producer, consumer)` and not one instruction wider. Only
    stalls issued at or after the producer separate it from the consumer; a
    cycle spent before the producer delays both equally and closes no gap at
    all. Spending outside the window looks like progress, terminates the search
    with a program that is still short, and is exactly the bug that produced a
    silently wrong fp64 result here.

    Fills the instruction nearest the consumer first and spills backwards, since
    a cycle anywhere inside the window counts the same and leaving the earlier
    ones free keeps room for their own consumers. Returns what could not be
    placed.

    Instructions already carrying the safe stall encoding are left alone: that
    encoding covers any dependency, and overwriting it with a finite number
    downgrades a guarantee into an estimate.
    """
    index = consumer - 1
    while amount > 0 and index >= producer:
        if index in pinned:
            # already carrying the safe encoding, which covers everything;
            # raising it would replace a guarantee with a number
            index -= 1
            continue
        room = STALL_MAX - stalls[index]
        if room > 0:
            give = min(room, amount)
            stalls[index] += give
            amount -= give
        index -= 1
    return amount


def schedule_program(
    program: Program | list[Instruction],
    model: LatencyModel,
    observed: ObservedStalls | None = None,
) -> ScheduleResult:
    """Rewrite every control word so the program's dependencies are covered.

    Works one basic block at a time. A value defined in another block is not
    assumed live, which is the same conservatism the verifier applies, and for
    the same reason: a linear listing is not a control-flow graph.
    """
    if not isinstance(program, Program):
        program = Program(instructions=list(program), labels={})

    count = len(program.instructions)
    pinned: set[int] = set()
    stalls = [1] * count
    waits = [0] * count
    write_barriers = [NO_BARRIER] * count
    result = ScheduleResult(words=[])

    cfg = build_cfg(program)

    # ---- scoreboards
    # allocated per block, then propagated along edges to a fixed point: a value
    # loaded before a loop and used inside it gets no wait from a block-local pass
    barrier_of: list[int] = [NO_BARRIER] * count

    for block in cfg.blocks:
        # barrier -> registers it guards; freed once a consumer has read them all
        protects: dict[int, set[RegRef]] = {}

        for index in range(block.start, block.end):
            instr = program.instructions[index]
            if instr.word is None:
                continue
            access = operand_access(instr.mnemonic, instr.operands)

            for sb in list(protects):
                if protects[sb] & access.real_uses:
                    del protects[sb]

            record = model.lookup(instr.opcode)
            if record.kind is LatencyClass.VARIABLE and access.real_defs:
                # lowest free index, so a kernel's scoreboards read in issue order.
                # inherited read barriers are not reserved: sharing a counter only
                # ever makes a wait cover more
                free = next((sb for sb in range(SCOREBOARDS) if sb not in protects), None)
                if free is not None:
                    protects[free] = set(access.real_defs)
                    result.scoreboards_used += 1
                else:
                    # all six are busy, so share the one guarding fewest registers;
                    # refusing instead rejected 45 of 317 kernels outright
                    free = min(protects, key=lambda sb: (len(protects[sb]), sb))
                    protects[free] |= set(access.real_defs)
                    result.scoreboards_shared += 1
                barrier_of[index] = free
                write_barriers[index] = free

    # register -> barriers that may still be outstanding for it on entry
    entry_state: list[dict[RegRef, frozenset[int]]] = [{} for _ in cfg.blocks]

    def transfer(block_index: int, state: dict[RegRef, frozenset[int]], emit: bool):
        """Walk a block, optionally recording the waits it needs."""
        live = dict(state)
        block = cfg.blocks[block_index]
        for index in range(block.start, block.end):
            instr = program.instructions[index]
            if instr.word is None:
                continue
            access = operand_access(instr.mnemonic, instr.operands)

            needed = 0
            for reg in access.real_uses:
                for sb in live.get(reg, ()):
                    needed |= 1 << sb

            if instr.opcode in _OPAQUE_TRANSFERS:
                # the callee may read anything, so wait on every live scoreboard
                # rather than only the ones named here
                for barriers in live.values():
                    for sb in barriers:
                        needed |= 1 << sb
            if emit:
                waits[index] |= needed

            if needed and not access.guard:
                # a predicated wait may not happen, so downstream cannot lean on it.
                # the checker stays lenient here because ptxas does lean on it
                cleared = {sb for sb in range(SCOREBOARDS) if (needed >> sb) & 1}
                live = {r: frozenset(bs - cleared) for r, bs in live.items()}

            barrier = barrier_of[index]
            mine = frozenset({barrier}) if barrier != NO_BARRIER else frozenset()
            for reg in access.real_defs:
                # a predicated write may not happen, so the earlier producer is
                # still outstanding: add rather than replace
                live[reg] = (live.get(reg, frozenset()) | mine) if access.guard else mine
        return live

    worklist = [0]
    # the lattice is finite, so this bound is a bug detector rather than a cutoff
    guard = 0
    while worklist and guard < MAX_PASSES * max(1, len(cfg.blocks)):
        guard += 1
        current = worklist.pop()
        out = transfer(current, entry_state[current], emit=False)
        for succ in cfg.blocks[current].successors:
            merged = dict(entry_state[succ])
            changed = False
            for reg, barriers in out.items():
                # union rather than intersect: a barrier outstanding on any path in
                union = merged.get(reg, frozenset()) | barriers
                if union != merged.get(reg):
                    merged[reg] = union
                    changed = True
            if changed:
                entry_state[succ] = merged
                if succ not in worklist:
                    worklist.append(succ)

    for block in cfg.blocks:
        transfer(block.index, entry_state[block.index], emit=True)

    # ---- stalls, to a fixed point
    for block in cfg.blocks:
        for _ in range(MAX_PASSES):
            result.passes += 1
            short = False

            last_def: dict[RegRef, list[_Producer]] = {}
            elapsed: dict[int, int] = {}

            for index in range(block.start, block.end):
                instr = program.instructions[index]
                if instr.word is None:
                    continue
                access = operand_access(instr.mnemonic, instr.operands)

                # a call reads registers its operand text never names, the return
                # address among them, so treat it as reading everything live
                opaque = instr.opcode in _OPAQUE_TRANSFERS
                consumed = set(last_def) if opaque else access.real_uses
                for reg in sorted(consumed, key=str):
                    for producer in last_def.get(reg, ()):
                        record = model.lookup(producer.opcode)
                        if producer.kind is LatencyClass.FIXED:
                            needed = _requirement(
                                producer.mnemonic,
                                instr.opcode,
                                record.cycles,
                                observed,
                                guard=reg == access.guard,
                            )
                        else:
                            # the wait covers the long part; this is the residue
                            needed = _scoreboarded_requirement(
                                producer.mnemonic,
                                instr.opcode,
                                observed,
                                guard=reg == access.guard,
                            )
                            if not needed:
                                continue
                        if opaque:
                            # mined pairings for a transfer describe what was live,
                            # not what was read, so use the producer's own latency
                            needed = max(needed, record.cycles)
                        have = elapsed.get(producer.index, 0)
                        if have >= needed:
                            continue

                        short = True
                        leftover = _place_stall(
                            stalls, pinned, producer.index, index, needed - have
                        )
                        result.stalls_added += (needed - have) - leftover
                        if leftover:
                            # covering this needs NOPs, which would move every
                            # address, so it is reported rather than half-done
                            note = (
                                f"#{producer.index} {producer.opcode} -> #{index} {instr.opcode}: "
                                f"{leftover} of {needed} cycles will not fit in the "
                                f"{index - producer.index} instruction window"
                            )
                            if note not in result.unplaceable:
                                result.unplaceable.append(note)
                            # window too small, so fall back to the safe encoding:
                            # nine times the cost and unconditionally correct
                            floor = _NEVER_ZERO_STALL.get(producer.opcode)
                            if floor is not None:
                                # cannot take the safe encoding, so it gets the
                                # smallest stall the vendor ever gives it
                                stalls[producer.index] = max(stalls[producer.index], floor)
                                continue
                            stalls[producer.index] = STALL_YIELD
                            pinned.add(producer.index)
                            if producer.index not in result.yielded:
                                result.yielded.append(producer.index)
                            continue

                charge = SATURATION if index in pinned else stalls[index]
                for key in elapsed:
                    elapsed[key] = min(SATURATION, elapsed[key] + charge)

                produced = _Producer(
                    index,
                    instr.opcode,
                    instr.mnemonic,
                    model.lookup(instr.opcode).kind,
                    write_barriers[index],
                )
                for reg in access.real_defs:
                    # a predicated write leaves the previous producer reachable,
                    # so it joins the list rather than replacing it
                    if access.guard:
                        last_def.setdefault(reg, []).append(produced)
                    else:
                        last_def[reg] = [produced]
                    elapsed[index] = stalls[index]

            if not short:
                break

    # ---- take back what nothing needs
    # the fixed point only ever adds, so stall placed for one pair can already
    # cover a later one; lowering is checked against the same requirement
    for block in cfg.blocks:
        for index in range(block.end - 1, block.start - 1, -1):
            if index in pinned or program.instructions[index].word is None:
                continue
            floor = _NEVER_ZERO_STALL.get(program.instructions[index].opcode, 1)
            while stalls[index] > floor:
                stalls[index] -= 1
                if _short_pair(block, program, stalls, pinned, model, observed):
                    stalls[index] += 1
                    break
                result.stalls_added -= 1

    # ---- anything crossing a block boundary
    # the safe stall encoding covers what leaves a block, and costs ~37 cycles,
    # so it goes only where liveness says something actually does
    live_out = _live_out(cfg, program)
    for block in cfg.blocks:
        last = block.end - 1
        if not (0 <= last < count) or last in pinned:
            continue
        if not live_out[block.index]:
            continue
        instr = program.instructions[last]
        floor = _NEVER_ZERO_STALL.get(instr.opcode if instr.word is not None else "")
        if floor is not None:
            stalls[last] = max(stalls[last], floor)
            continue
        stalls[last] = STALL_YIELD
        pinned.add(last)
        result.yielded.append(last)

    # ---- windows a read barrier covers
    # one barrier stands in for a whole run of reads, so the vendor's gaps
    # inside the window are a floor rather than something to compress (finding 13)
    block_of_index: dict[int, int] = {}
    for block in cfg.blocks:
        for index in range(block.start, block.end):
            block_of_index[index] = block.start

    previous_barrier = -1
    for index, instr in enumerate(program.instructions):
        if instr.word is None or instr.word.field("read_barrier") == NO_BARRIER:
            continue
        start = max(block_of_index.get(index, 0), previous_barrier + 1)
        for covered in range(start, index + 1):
            original = program.instructions[covered]
            if original.word is not None:
                stalls[covered] = max(stalls[covered], original.word.field("stall"))
        previous_barrier = index

    # read barriers keep the vendor's numbering, so waits on them are kept too
    read_barrier_mask = 0
    for instr in program.instructions:
        if instr.word is None:
            continue
        barrier = instr.word.field("read_barrier")
        if barrier != NO_BARRIER:
            read_barrier_mask |= 1 << barrier

    # ---- emit
    for index, instr in enumerate(program.instructions):
        if instr.word is None:
            result.words.append(Word(0))
            continue
        word = instr.word

        # inherited, not recomputed: no measured model of a late operand read
        inherited_read_barrier = word.field("read_barrier")
        wait = waits[index] | (word.field("wait_mask") & read_barrier_mask)

        word = word.with_field("stall", stalls[index])
        # the yield bit tracks the stall; other pairings are words the vendor
        # never emits and nvdisasm refuses while the GPU runs them anyway
        word = word.with_field("yield_", 1 if stalls[index] == 1 else 0)
        word = word.with_field("wait_mask", wait)
        word = word.with_field("write_barrier", write_barriers[index])
        word = word.with_field("read_barrier", inherited_read_barrier)
        # a reuse flag from a different schedule is a hazard, so it is cleared
        word = word.with_field("reuse", 0)
        result.words.append(word)

    return result


def issue_cycles(words, instructions=None) -> int:
    """Cycles a schedule spends issuing, counting only what runs.

    Everything after the first `EXIT` is padding the assembler emits to fill a
    cache line and never issues, so counting it makes a kernel with more padding
    look more expensive. Ignoring that is how a first attempt at this measured
    basalt as nearly four times *faster* than the vendor.

    A cost model rather than a measurement: it counts what the control bits ask
    the scheduler to wait, which is the thing basalt decides, and says nothing
    about memory or occupancy.
    """
    reachable = len(words)
    source = instructions if instructions is not None else []
    for index, instruction in enumerate(source):
        if instruction.word is not None and instruction.opcode == "EXIT":
            reachable = index + 1
            break

    total = 0
    for index in range(min(reachable, len(words))):
        word = words[index]
        if word is None:
            continue
        if instructions is not None and instructions[index].word is None:
            continue
        stall = word.field("stall")
        total += YIELD_COST if stall == STALL_YIELD else stall
    return total
