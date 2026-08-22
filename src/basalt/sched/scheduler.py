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

# What the safe stall encoding costs when counting issue cycles. A zero stall is
# a long wait rather than no wait, measured at about 37 cycles per instruction
# (see docs/FINDINGS.md), so costing it as zero would flatter a schedule that
# leans on it, which is exactly the schedule basalt produces.
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

# Control transfers to somewhere this analysis has not followed. What runs next
# may use any register, so nothing may still be in flight when one of these
# issues. Indirect branches are here for the same reason a call is: the
# destination is computed and the control-flow graph says so.
_OPAQUE_TRANSFERS = frozenset({"CALL", "RET", "BRX", "JMX", "RTT", "BPT"})

# Instructions `ptxas` never gives the zero-stall encoding to, with the smallest
# stall it is seen using for each. Measured over the whole corpus: `EXIT` 0 times
# out of 329, `RET` 0 of 5, `CALL` 0 of 5, `BAR` 0 of 3. `BRA` is deliberately
# not here, since it takes a zero stall 329 times, so this is a property of
# these instructions rather than of control transfer in general.
#
# It matters because the safe stall encoding is basalt's fallback everywhere. It
# covers any dependency, so it gets reached for at block boundaries and wherever
# a requirement will not fit, and reaching for the safe answer on an instruction
# the vendor never applies it to lands outside what the encoding supports.
_NEVER_ZERO_STALL: dict[str, int] = {"EXIT": 5, "RET": 5, "CALL": 5, "BAR": 6}


@dataclass
class ScheduleResult:
    """A rescheduled program, and what could not be arranged."""

    words: list[Word]
    passes: int = 0
    stalls_added: int = 0
    scoreboards_used: int = 0
    # Producers that had to share an already-busy scoreboard because all six
    # were guarding something. Correct but slightly over-synchronised, so it is
    # counted rather than hidden.
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
    # Allocation is per block, which is fine: a scoreboard is a counter, so
    # reusing an index across blocks makes a wait cover more than it needs to
    # rather than less. The waits themselves cannot be per block, though. A
    # value loaded before a loop and consumed inside it is the ordinary shape of
    # real code, and a block-local analysis starts with an empty map and emits
    # no wait at all for it, which is silently wrong.
    #
    # So allocation runs per block and then the outstanding barriers are
    # propagated along control-flow edges to a fixed point, exactly as the
    # verifier propagates reaching definitions, and the waits are emitted from
    # the settled state.
    barrier_of: list[int] = [NO_BARRIER] * count

    for block in cfg.blocks:
        # barrier -> the registers it is currently protecting. A barrier is free
        # again once something has consumed every register it was guarding,
        # because the wait that consumer will carry is what releases it. Without
        # this the six scoreboards are exhausted by the seventh load in a block.
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
                free = next((sb for sb in range(SCOREBOARDS) if sb not in protects), None)
                if free is not None:
                    protects[free] = set(access.real_defs)
                    result.scoreboards_used += 1
                else:
                    # Every scoreboard is already guarding something, so share
                    # one. A scoreboard is a counter rather than a flag: several
                    # producers can signal the same index, and a wait on it
                    # blocks until all of them have reported. Sharing therefore
                    # makes the wait cover more than it strictly needs to, which
                    # costs a little parallelism and cannot be unsafe.
                    #
                    # Not an edge case. Six loads in flight is ordinary in a
                    # tensor kernel, and refusing to schedule the seventh
                    # rejected 45 of the 317 corpus kernels outright. The vendor
                    # compiler shares for the same reason.
                    #
                    # The one holding the fewest registers is chosen, so the
                    # extra waiting lands where the fewest consumers will feel
                    # it.
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
                # Control is about to leave for code this analysis has not read,
                # and the callee is free to use any register. Everything still
                # outstanding has to have landed first, so every live scoreboard
                # is waited on rather than only the ones this instruction reads.
                #
                # `ptxas` does the same: the `CALL.REL.NOINC` in the 4-bit MMA
                # kernels waits on two scoreboards it has no operand interest in.
                # basalt waited on none and the kernel came out non-deterministic.
                for barriers in live.values():
                    for sb in barriers:
                        needed |= 1 << sb
            if emit:
                waits[index] |= needed

            if needed and not access.guard:
                # Waiting clears those scoreboards for everything downstream,
                # but only when the wait is certain to happen. A predicated
                # instruction may not, so a later consumer that leans on its
                # wait can read a result that never landed.
                #
                # Measured: `MUFU.EX2` feeding a store through a predicated
                # `FMUL` is wrong every time basalt leaves the store without its
                # own wait, and right as soon as it has one. Same for `SQRT` and
                # `RSQ`. Not relying on it costs one extra wait.
                #
                # The checker deliberately does not apply this rule. `ptxas`
                # does lean on predicated waits and its output runs correctly,
                # so calling that an error would be a false positive against the
                # reference. Being stricter about what basalt emits than about
                # what it accepts is the right way round.
                cleared = {sb for sb in range(SCOREBOARDS) if (needed >> sb) & 1}
                live = {r: frozenset(bs - cleared) for r, bs in live.items()}

            barrier = barrier_of[index]
            mine = frozenset({barrier}) if barrier != NO_BARRIER else frozenset()
            for reg in access.real_defs:
                # a predicated write may not happen, so whatever was outstanding
                # for that register is still outstanding and still needs waiting
                # on. replacing here instead of adding drops the wait on the
                # earlier producer entirely.
                live[reg] = (live.get(reg, frozenset()) | mine) if access.guard else mine
        return live

    worklist = [0]
    guard = 0
    while worklist and guard < MAX_PASSES * max(1, len(cfg.blocks)):
        guard += 1
        current = worklist.pop()
        out = transfer(current, entry_state[current], emit=False)
        for succ in cfg.blocks[current].successors:
            merged = dict(entry_state[succ])
            changed = False
            for reg, barriers in out.items():
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

                for reg in sorted(access.real_uses, key=str):
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
                            # A variable-latency producer is covered by the wait for
                            # the long part of its result, and by a small gap for
                            # the rest. Only the second part is scheduled here; the
                            # wait itself was assigned above.
                            needed = _scoreboarded_requirement(
                                producer.mnemonic,
                                instr.opcode,
                                observed,
                                guard=reg == access.guard,
                            )
                            if not needed:
                                continue
                        have = elapsed.get(producer.index, 0)
                        if have >= needed:
                            continue

                        short = True
                        leftover = _place_stall(
                            stalls, pinned, producer.index, index, needed - have
                        )
                        result.stalls_added += (needed - have) - leftover
                        if leftover:
                            # the window between producer and consumer cannot hold
                            # the requirement. covering it needs NOPs inserted, which
                            # would change instruction addresses, so it is reported
                            # rather than half-done.
                            note = (
                                f"#{producer.index} {producer.opcode} -> #{index} {instr.opcode}: "
                                f"{leftover} of {needed} cycles will not fit in the "
                                f"{index - producer.index} instruction window"
                            )
                            if note not in result.unplaceable:
                                result.unplaceable.append(note)
                            # The window cannot hold the requirement, so fall back to
                            # the safe stall encoding on the producer. A zero stall
                            # waits for outstanding results as well as elapsed cycles
                            # (see docs/FINDINGS.md), which costs about nine times a
                            # scheduled instruction and is unconditionally correct.
                            # Being slow is a trade; being wrong is not.
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

    # ---- anything crossing a block boundary
    # The analysis above is per block, so a value defined in one block and
    # consumed in another gets no coverage from it. Rather than guess which
    # values are live out, the last instruction of every block is given the safe
    # stall encoding, which covers any dependency that leaves the block. That is
    # blunt and it is correct, and correctness is the one thing this cannot
    # trade away.
    for block in cfg.blocks:
        last = block.end - 1
        if 0 <= last < count and last not in pinned:
            instr = program.instructions[last]
            floor = _NEVER_ZERO_STALL.get(instr.opcode if instr.word is not None else "")
            if floor is not None:
                stalls[last] = max(stalls[last], floor)
                continue
            stalls[last] = STALL_YIELD
            pinned.add(last)
            result.yielded.append(last)

    # ---- emit
    for index, instr in enumerate(program.instructions):
        if instr.word is None:
            result.words.append(Word(0))
            continue
        word = instr.word

        # Read barriers are carried over rather than recomputed. They guard an
        # instruction that consumes its sources late, and basalt has no measured
        # model for how long that takes: FINDINGS records it as the one hazard
        # class reported as a warning for exactly this reason. Inventing a value
        # would be worse than preserving one that is known to work, so the
        # original barrier is kept and the waits that serviced it are folded into
        # the new wait mask. Everything else here is computed from scratch.
        inherited_read_barrier = word.field("read_barrier")
        wait = waits[index]
        if inherited_read_barrier != NO_BARRIER or word.field("read_barrier") != NO_BARRIER:
            wait |= word.field("wait_mask")

        word = word.with_field("stall", stalls[index])
        # The yield bit is not independent of the stall. Across the whole corpus
        # `ptxas` emits a zero stall with the bit clear 4205 times and set never,
        # and a stall of one with it set 1123 times and clear never. Writing a
        # stall without the matching bit produces a combination the vendor never
        # emits, and `nvdisasm` rejects it outright: "undefined value 0x10 for
        # table TABLES_opex_0". The GPU runs it anyway, which is worse rather
        # than better, because it means nothing complains until something tries
        # to read the result back.
        #
        # Clearing the bit at any other stall is a throughput choice rather than
        # a correctness one, and `ptxas` is seen doing it at every stall value
        # from 2 up, so every pair this produces is one the vendor also emits.
        word = word.with_field("yield_", 1 if stalls[index] == 1 else 0)
        word = word.with_field("wait_mask", wait)
        word = word.with_field("write_barrier", write_barriers[index])
        word = word.with_field("read_barrier", inherited_read_barrier)
        # the reuse cache is an optimisation rather than a correctness mechanism,
        # and a reuse flag left over from a different schedule is a hazard, so it
        # is cleared instead of carried
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
