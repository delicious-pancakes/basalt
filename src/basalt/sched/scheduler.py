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

import re
from dataclasses import dataclass, field

from ..disasm import Instruction, Program
from ..encoding import NO_BARRIER, STALL_YIELD, Word
from ..verify.cfg import build_cfg
from ..verify.latency import (
    GUARD_CYCLES,
    SCOREBOARD_RESIDUE_CYCLES,
    LatencyClass,
    LatencyModel,
)
from ..verify.observed import ObservedStalls, anti_dependency_cycles
from ..verify.operands import RegRef, operand_access

__all__ = [
    "SCOREBOARDS",
    "SCOREBOARD_OPERAND",
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

# stalls ptxas pairs with the yield hint, fitted rather than reasoned: 93.7% of
# 37,008 instructions against 73.2% for the guess it replaced (finding 26)
YIELD_STALL_RANGE = (1, 12)

# `DEPBAR.LE SB0, 0x0` waits by naming its scoreboard, so renumbering the
# signaller unpairs them with nothing in the encoding to show it (finding 27)
SCOREBOARD_OPERAND = re.compile(r"\bSB(\d)\b")

# no register result, so no latency class, but the operand read is still in
# flight after they issue: the vendor guards these with a read barrier too
_LATE_READING_CONTROL = frozenset({"STG", "STS", "STL", "ST", "RED"})


@dataclass
class ScheduleResult:
    """A rescheduled program, and what could not be arranged."""

    words: list[Word]
    passes: int = 0
    stalls_added: int = 0
    scoreboards_used: int = 0
    # shared an already-busy scoreboard: over-synchronised, so counted not hidden
    scoreboards_shared: int = 0
    read_barriers_used: int = 0
    # what this computed a word for. the rest keeps the vendor's, and saying
    # which is the difference between standing behind a schedule and copying it
    analysed: set[int] = field(default_factory=set)
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
            f"{self.scoreboards_shared} shared, {self.read_barriers_used} read barriers, "
            f"{len(self.yielded)} safe stalls: {verdict}"
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
    floor = GUARD_CYCLES if guard else cycles
    if observed is not None:
        evidence = observed.requirement(producer, key)
        if evidence is not None:
            # `emit` and not `minimum`: a wider body of code may raise this floor
            # and never lower it, since a schedule has to be right, not defensible
            return max(evidence.emit, floor)
    return floor


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
    if evidence is None:
        return 0
    return max(evidence.emit, SCOREBOARD_RESIDUE_CYCLES)


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


def _defs(program, index: int) -> frozenset[RegRef]:
    """Registers one instruction writes, or nothing when it did not decode."""
    instr = program.instructions[index]
    if instr.word is None:
        return frozenset()
    return operand_access(instr.mnemonic, instr.operands).real_defs


def _late_reader(opcode: str, model: LatencyModel) -> bool:
    """Does this instruction still hold its source registers after it issues?

    Variable latency is the signal. Something that answers on a fixed schedule
    has taken its operands by a fixed point too, and the anti-dependency stall
    covers that; something the hardware makes you wait for has not. Every one of
    the 299 read barriers in the corpus is on an instruction in one of these two
    sets, and nothing outside them ever carries one.
    """
    return model.lookup(opcode).kind is LatencyClass.VARIABLE or opcode in _LATE_READING_CONTROL


def _read_barrier_windows(cfg, program, model, waits, write_barriers) -> list[tuple[int, int]]:
    """`(reader, writer)` pairs where an operand read outlives its register.

    A read barrier is in order: everything issued before the setter has finished
    reading by the time it clears. So one barrier on the *last* late reader
    before an overwrite covers every earlier reader too, which is why the vendor
    emits 299 of them across 37,008 instructions rather than one per load.

    Nothing is needed where the overwriter already waits on the reader's write
    barrier, since that wait cannot clear before the read has happened. That one
    exemption accounts for 246 of the 318 overwrites the vendor leaves bare.

    A forwards fixed point over the graph rather than a walk through each block,
    because the case that matters most is loop-carried: a tiled matmul stores to
    shared memory and the next iteration overwrites the register it stored from,
    and a per-block scan sees an outstanding read at the end of the body and no
    overwrite anywhere. Every read barrier that kernel needs is on that edge.
    """
    entry: list[dict[RegRef, int]] = [{} for _ in cfg.blocks]
    windows: set[tuple[int, int]] = set()

    def walk(block, state: dict[RegRef, int], collect: bool) -> dict[RegRef, int]:
        latest = dict(state)
        for index in range(block.start, block.end):
            instr = program.instructions[index]
            if instr.word is None:
                continue
            access = operand_access(instr.mnemonic, instr.operands)

            reader = max((latest[reg] for reg in access.real_defs if reg in latest), default=None)
            # an instruction that reads and writes the same register is not
            # racing itself, so its own entry never counts
            if collect and reader is not None and reader != index:
                # basalt's own number, not the vendor's: `waits` is basalt's, and
                # reading one against the other is what finding 24 was about
                barrier = write_barriers[reader]
                if not (barrier != NO_BARRIER and (waits[index] >> barrier) & 1):
                    windows.add((reader, index))
            for reg in access.real_defs:
                latest.pop(reg, None)

            if _late_reader(instr.opcode, model):
                for reg in access.real_uses:
                    latest[reg] = index
        return latest

    worklist = [0] if cfg.blocks else []
    guard = 0
    while worklist and guard < MAX_PASSES * max(1, len(cfg.blocks)):
        guard += 1
        current = worklist.pop()
        out = walk(cfg.blocks[current], entry[current], collect=False)
        for succ in cfg.blocks[current].successors:
            merged = dict(entry[succ])
            changed = False
            for reg, reader in out.items():
                # the latest reader wins: its barrier covers every earlier one
                if merged.get(reg, -1) < reader:
                    merged[reg] = reader
                    changed = True
            if changed:
                entry[succ] = merged
                if succ not in worklist:
                    worklist.append(succ)

    for block in cfg.blocks:
        walk(block, entry[block.index], collect=True)
    return sorted(windows)


def _assign_read_barriers(
    windows, write_barriers, waits, count
) -> tuple[list[int], list[int], int]:
    """Give each window a scoreboard, and the waits that go with it.

    Read and write barriers share one six-entry space, so a number is only free
    over a window if no write barrier is outstanding across it. Windows that
    end at the same instruction are merged onto the later reader first, since
    that is the one whose barrier covers the rest.
    """
    busy = [0] * count
    for setter, barrier in enumerate(write_barriers):
        if barrier == NO_BARRIER:
            continue
        # busy until the *next* wait, not the last one anywhere: a scoreboard
        # reused four times would otherwise read as occupied for the whole kernel
        last = next((i for i in range(setter + 1, count) if (waits[i] >> barrier) & 1), count - 1)
        for i in range(setter, last + 1):
            busy[i] |= 1 << barrier

    read_barrier_of = [NO_BARRIER] * count
    extra = [0] * count
    shared = 0
    for reader, writer in sorted(windows, key=lambda w: (w[1], -w[0])):
        # a loop-carried window runs from the reader to the end of the body and
        # again from the head to the writer, so the whole enclosing span is taken
        span = range(min(reader, writer), max(reader, writer) + 1)
        if read_barrier_of[reader] != NO_BARRIER:
            # a reader can cover two overwrites, so the number is busy to the
            # later one; marking only the first let a second reuse it
            already = read_barrier_of[reader]
            extra[writer] |= 1 << already
            for i in span:
                busy[i] |= 1 << already
            continue
        # never one the reader waits on itself: signalling a barrier the same
        # instruction waits on is waiting for a read that has not started
        taken = waits[reader] | extra[reader]
        for i in span:
            taken |= busy[i]
        free = next((sb for sb in range(SCOREBOARDS) if not (taken >> sb) & 1), None)
        if free is None:
            # every number is taken. a scoreboard is a counter, so sharing waits
            # for both signals: over-synchronised, counted, never a stale read
            mine = waits[reader] | extra[reader]
            choices = [sb for sb in range(SCOREBOARDS) if not (mine >> sb) & 1] or list(
                range(SCOREBOARDS)
            )
            free = min(choices, key=lambda sb: sum((busy[i] >> sb) & 1 for i in span))
            shared += 1
        read_barrier_of[reader] = free
        extra[writer] |= 1 << free
        for i in span:
            busy[i] |= 1 << free
    return read_barrier_of, extra, shared


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

    # ---- scoreboards named in an operand
    # pinned to the vendor's number, which is off limits until the namer has run
    pinned_barrier: dict[int, int] = {}
    reserved: list[tuple[int, int, int]] = []
    for index, instr in enumerate(program.instructions):
        if instr.word is None:
            continue
        for match in SCOREBOARD_OPERAND.finditer(instr.operands):
            sb = int(match.group(1))
            signaller = next(
                (
                    earlier
                    for earlier in range(index - 1, -1, -1)
                    if (word := program.instructions[earlier].word) is not None
                    and word.field("write_barrier") == sb
                ),
                None,
            )
            if signaller is not None:
                pinned_barrier[signaller] = sb
                reserved.append((sb, signaller, index))

    # ---- scoreboards
    # per block, then propagated to a fixed point: a value loaded before a loop
    # and used inside it gets no wait from a block-local pass
    barrier_of: list[int] = [NO_BARRIER] * count

    def unavailable(index: int) -> set[int]:
        """Scoreboards a named-operand pairing is using across this instruction."""
        return {sb for sb, start, end in reserved if start <= index <= end}

    def allocate() -> None:
        for i in range(count):
            barrier_of[i] = NO_BARRIER
            write_barriers[i] = NO_BARRIER
        for block in cfg.blocks:
            # barrier -> registers it guards; freed once a consumer has read them all
            protects: dict[int, set[RegRef]] = {}

            for index in range(block.start, block.end):
                instr = program.instructions[index]
                if instr.word is None:
                    continue
                access = operand_access(instr.mnemonic, instr.operands)

                # freed after this instruction takes its own, never before: a
                # barrier signalled by the instruction waiting on it is finding 24
                released = [sb for sb in protects if protects[sb] & access.real_uses]

                if index in pinned_barrier:
                    barrier_of[index] = write_barriers[index] = pinned_barrier[index]
                    for sb in released:
                        del protects[sb]
                    continue

                record = model.lookup(instr.opcode)
                if record.kind is LatencyClass.VARIABLE and access.real_defs:
                    # lowest free index, so a kernel's scoreboards read in issue order
                    avoid = unavailable(index)
                    free = next(
                        (sb for sb in range(SCOREBOARDS) if sb not in protects and sb not in avoid),
                        None,
                    )
                    if free is not None:
                        protects[free] = set(access.real_defs)
                        result.scoreboards_used += 1
                    else:
                        # all six are busy, so share the one guarding fewest registers;
                        # refusing instead rejected 45 of 317 kernels outright
                        candidates = {sb for sb in protects if sb not in avoid} or (
                            set(protects) - unavailable(index)
                        )
                        if candidates:
                            free = min(candidates, key=lambda sb: (len(protects[sb]), sb))
                            protects[free] |= set(access.real_defs)
                        else:
                            # nothing in use and every number ruled out, so the
                            # repair has run out of room and this round is the last
                            free = 0
                            protects[free] = set(access.real_defs)
                        result.scoreboards_shared += 1
                    barrier_of[index] = free
                    write_barriers[index] = free
                for sb in released:
                    if sb in protects and sb != barrier_of[index]:
                        del protects[sb]

    # register -> producers still outstanding on entry. keyed on the producer,
    # since a number is reused and a wait on the second use says nothing
    entry_state: list[dict[RegRef, frozenset[int]]] = [{} for _ in cfg.blocks]
    # producers some consumer actually waits for; the rest are signalling into
    # a void and are taking a scoreboard a read barrier could have had
    credited: set[int] = set()

    def transfer(block_index: int, state: dict[RegRef, frozenset[int]], emit: bool):
        """Walk a block, optionally recording the waits it needs."""
        live = dict(state)
        block = cfg.blocks[block_index]
        for index in range(block.start, block.end):
            instr = program.instructions[index]
            if instr.word is None:
                continue
            access = operand_access(instr.mnemonic, instr.operands)

            wanted: set[int] = set()
            for reg in access.real_uses:
                wanted |= set(live.get(reg, ()))

            if instr.opcode in _OPAQUE_TRANSFERS:
                # the callee may read anything, so wait on every live scoreboard
                # rather than only the ones named here
                for producers in live.values():
                    wanted |= set(producers)

            needed = 0
            for producer in wanted:
                needed |= 1 << barrier_of[producer]
            if emit:
                waits[index] |= needed
                # here as well as on being cleared below: a guarded consumer
                # emits its wait and clears nothing
                credited.update(wanted)

            if needed and not access.guard:
                # a predicated wait may not happen, so downstream cannot lean on it.
                # the checker stays lenient here because ptxas does lean on it
                covered = {q for ps in live.values() for q in ps if (needed >> barrier_of[q]) & 1}
                live = {r: frozenset(ps - covered) for r, ps in live.items()}
                if emit:
                    # on being cleared, not on being asked for: a wait drains
                    # every producer sharing the number and downstream leans on it
                    credited.update(covered)

            mine = frozenset({index}) if barrier_of[index] != NO_BARRIER else frozenset()
            for reg in access.real_defs:
                # a predicated write may not happen, so the earlier producer is
                # still outstanding: add rather than replace
                live[reg] = (live.get(reg, frozenset()) | mine) if access.guard else mine
        return live

    def converge() -> None:
        for i in range(count):
            waits[i] = 0
        for state in entry_state:
            state.clear()
        credited.clear()
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
                for reg, producers in out.items():
                    # union rather than intersect: a producer outstanding on any path in
                    union = merged.get(reg, frozenset()) | producers
                    if union != merged.get(reg):
                        merged[reg] = union
                        changed = True
                if changed:
                    entry_state[succ] = merged
                    if succ not in worklist:
                        worklist.append(succ)

        for block in cfg.blocks:
            transfer(block.index, entry_state[block.index], emit=True)

    # waiting on the number about to be signalled is reuse, not self-reference:
    # ptxas does it 251 times in 37,008 instructions (finding 24)
    allocate()
    converge()

    # ---- barriers nothing waits for
    # signalled and never waited on is not synchronisation, it is a number a
    # read barrier could have had; the allocator runs before the dataflow
    live_out = _live_out(cfg, program)
    block_of = {i: b.index for b in cfg.blocks for i in range(b.start, b.end)}

    def uncredited(i: int) -> bool:
        # a pinned barrier is waited on by name in a later operand, which the
        # dataflow never sees, so it is never credited and must not be dropped
        return (
            i in block_of
            and write_barriers[i] != NO_BARRIER
            and i not in credited
            and i not in pinned_barrier
            and not _defs(program, i) & live_out[block_of[i]]
        )

    if any(uncredited(i) for i in range(count)):
        for index in range(count):
            if not uncredited(index):
                continue
            write_barriers[index] = NO_BARRIER
            barrier_of[index] = NO_BARRIER
            result.scoreboards_used -= 1
        converge()

    # ---- read barriers
    # computed rather than inherited, so a program that never had one gets them
    windows = _read_barrier_windows(cfg, program, model, waits, write_barriers)
    read_barriers, read_waits, shared = _assign_read_barriers(windows, write_barriers, waits, count)
    for index in range(count):
        waits[index] |= read_waits[index]
    result.read_barriers_used = sum(1 for b in read_barriers if b != NO_BARRIER)
    result.scoreboards_shared += shared

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
    # the fixed point only adds, so stall placed for one pair can cover a later
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
    # the safe encoding costs ~37 cycles, so it goes only where liveness says
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
    # one barrier stands in for a run of reads, and that holds at the rate the
    # unit accepts work: `LDG` after `LDG` is 4 cycles over 1,953 observations
    block_of_index: dict[int, int] = {}
    for block in cfg.blocks:
        for index in range(block.start, block.end):
            block_of_index[index] = block.start

    previous_barrier = -1
    for index, instr in enumerate(program.instructions):
        if instr.word is None or read_barriers[index] == NO_BARRIER:
            continue
        start = max(block_of_index.get(index, 0), previous_barrier + 1)
        for covered in range(start, index):
            first = program.instructions[covered]
            second = program.instructions[covered + 1]
            if first.word is None or second.word is None or observed is None:
                continue
            rate = observed.issue_minimum(first.mnemonic, second.mnemonic)
            if stalls[covered] != STALL_YIELD:
                stalls[covered] = max(stalls[covered], rate)
        previous_barrier = index

    # ---- anti-dependencies
    # a read still in flight when its register is overwritten sees the new value
    for block in cfg.blocks:
        last_read: dict[RegRef, int] = {}
        for index in range(block.start, block.end):
            instr = program.instructions[index]
            if instr.word is None:
                continue
            access = operand_access(instr.mnemonic, instr.operands)
            for register in access.real_defs:
                reader = last_read.get(register)
                if reader is None or reader == index:
                    continue
                # a read barrier already covers this one, and the wait for it is
                # not a number of cycles
                if read_barriers[reader] != NO_BARRIER:
                    continue
                needed = anti_dependency_cycles(
                    program.instructions[reader].mnemonic, instr.opcode, observed
                )
                gap = sum(
                    SATURATION if stalls[i] == STALL_YIELD else stalls[i]
                    for i in range(reader, index)
                )
                if gap < needed:
                    stalls[reader] += needed - gap
            for register in access.real_uses:
                last_read[register] = index

    # ---- floors the dependency graph cannot see
    # the second `bar.sync` in a tiled loop guards shared memory, not a register,
    # and no def-use edge shows that
    for index, instr in enumerate(program.instructions):
        if instr.word is None or stalls[index] == STALL_YIELD:
            continue
        floor = _NEVER_ZERO_STALL.get(instr.opcode)
        if floor is not None:
            stalls[index] = max(stalls[index], floor)

    # a cubin holds more than its entry function: 125 of `mma.b1`'s 144
    # instructions sit in bodies the dataflow never reaches, so they keep theirs
    reachable: set[int] = set()
    stack = [0] if cfg.blocks else []
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(cfg.blocks[current].successors)
    result.analysed = {
        index
        for block in cfg.blocks
        if block.index in reachable
        for index in range(block.start, block.end)
        if program.instructions[index].word is not None
    }

    # ---- emit
    for index, instr in enumerate(program.instructions):
        if instr.word is None:
            result.words.append(Word(0))
            continue
        if index not in result.analysed:
            result.words.append(instr.word)
            continue
        word = instr.word

        wait = waits[index]

        word = word.with_field("stall", stalls[index])
        low, high = YIELD_STALL_RANGE
        word = word.with_field("yield_", int(low <= stalls[index] < high))
        word = word.with_field("wait_mask", wait)
        word = word.with_field("write_barrier", write_barriers[index])
        word = word.with_field("read_barrier", read_barriers[index])
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
