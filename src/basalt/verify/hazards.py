# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Checking that a program's control bits actually cover its data dependencies.

The rules the hardware does not enforce, stated precisely:

*Fixed-latency results.* An instruction at index `i` with latency `L` makes its
result available `L` cycles after it issues. The `stall` field of instruction
`k` is how many cycles pass before instruction `k+1` issues, so the elapsed time
between issuing `i` and issuing a consumer `j` is `sum(stall[i..j-1])`. The
consumer is safe only if that sum is at least `L` on *every* path that can reach
it. Nothing checks this at run time, which is why a violation is a wrong answer
rather than a crash.

*Variable-latency results.* Memory and special-register reads finish whenever
they finish, so they signal a scoreboard through `write_barrier` and consumers
block on it through `wait_mask`. Scoreboards are counters rather than flags:
several producers may signal the same one, and a single wait covers every
outstanding signal on it, for everything downstream.

*Operands still being read.* An instruction that consumes its sources late
signals a `read_barrier`. Anything overwriting those registers must wait on it,
or the reader sees a value written after it was scheduled to read.

Analysis runs over the whole control-flow graph, so a value defined before a
loop and consumed inside it is checked, as is one defined in one arm of a branch
and consumed after the join. Findings are emitted only once the dataflow has
settled, so convergence cannot report the same hazard twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ..disasm import Instruction, Program
from ..encoding import STALL_YIELD, effective_stall
from .cfg import ControlFlowGraph, build_cfg
from .flow import FlowState
from .latency import (
    ANTI_DEPENDENCY_CYCLES,
    GUARD_CYCLES,
    SCOREBOARD_RESIDUE_CYCLES,
    Confidence,
    LatencyClass,
    LatencyModel,
    LatencyRecord,
)
from .observed import MIN_OBSERVATIONS, ObservedStalls, anti_dependency_cycles
from .operands import RegRef, operand_access

__all__ = [
    "Hazard",
    "HazardKind",
    "Severity",
    "VerificationReport",
    "split_blocks",
    "verify_program",
]

NO_BARRIER = 7

# The lattice is finite, so this is a backstop against a bug rather than a limit
# the analysis is expected to reach.
MAX_ITERATIONS = 10_000


class HazardKind(StrEnum):
    UNDERSTALLED = "understalled"
    NO_BARRIER_SET = "no-barrier-set"
    BARRIER_NOT_AWAITED = "barrier-not-awaited"
    OVERWRITTEN_BEFORE_READ = "overwritten-before-read"


class Severity(StrEnum):
    ERROR = "error"  # a real hazard under the model in use
    WARNING = "warning"  # a hazard only under an assumed latency number
    INFO = "info"  # worth seeing, not necessarily wrong


@dataclass(frozen=True, slots=True)
class Hazard:
    """One place the control bits do not cover a dependency."""

    kind: HazardKind
    severity: Severity
    confidence: Confidence
    register: str
    def_index: int
    use_index: int
    def_text: str
    use_text: str
    required: int = 0
    actual: int = 0
    detail: str = ""

    @property
    def key(self) -> tuple:
        """Identity used to report a hazard once, however many paths reach it."""
        return (self.kind, self.register, self.def_index, self.use_index)

    def describe(self) -> str:
        where = f"#{self.def_index} -> #{self.use_index}"
        if self.kind is HazardKind.UNDERSTALLED:
            gap = f"{self.actual} of {self.required} cycles"
            return f"[{self.severity}] {where} {self.register}: {gap}. {self.detail}".rstrip()
        return f"[{self.severity}] {where} {self.register}: {self.detail}"


@dataclass
class VerificationReport:
    """Everything one verification run found, plus what it was checked against."""

    hazards: list[Hazard] = field(default_factory=list)
    instructions: int = 0
    blocks: int = 0
    checked_pairs: int = 0
    model_confidence: Confidence = Confidence.ASSUMED
    model_sku: str = ""
    unknown_opcodes: set[str] = field(default_factory=set)
    cross_block: bool = False
    incomplete_graph: bool = False

    @property
    def ok(self) -> bool:
        return not any(h.severity is Severity.ERROR for h in self.hazards)

    def by_severity(self, severity: Severity) -> list[Hazard]:
        return [h for h in self.hazards if h.severity is severity]

    def summary(self) -> str:
        errors = len(self.by_severity(Severity.ERROR))
        warnings = len(self.by_severity(Severity.WARNING))
        verdict = "clean" if not self.hazards else f"{errors} errors, {warnings} warnings"
        scope = "across blocks" if self.cross_block else "per block"
        return (
            f"{self.instructions} instructions in {self.blocks} blocks, "
            f"{self.checked_pairs} dependencies checked {scope}: {verdict}"
        )


def _exclusive(producer: str, consumer: str) -> bool:
    """True when two instructions are guarded by opposite halves of one predicate.

    `@P0 LDG` never feeds `@!P0 PRMT`: within a thread exactly one of them runs,
    so the register the load would have written is not the one the permute reads.
    """
    first, second = _guard(producer), _guard(consumer)
    return (
        first is not None and second is not None and first[0] == second[0] and first[1] != second[1]
    )


def _guard(text: str) -> tuple[str, bool] | None:
    """The guarding predicate and whether it is inverted, from instruction text."""
    token = text.split(maxsplit=1)[0] if text else ""
    if not token.startswith("@"):
        return None
    token = token[1:]
    inverted = token.startswith("!")
    return (token.lstrip("!"), inverted)


def _add(report: VerificationReport | None, seen: set[tuple] | None, hazard: Hazard) -> None:
    # callers only reach here on the recording pass; taking the optional rather
    # than asserting keeps the convergence pass free of a branch it never takes
    if report is None or seen is None:
        return
    if hazard.key not in seen:
        seen.add(hazard.key)
        report.hazards.append(hazard)


def _check_instruction(
    program: Program,
    index: int,
    state: FlowState,
    model: LatencyModel,
    report: VerificationReport | None,
    seen: set[tuple] | None,
    observed: ObservedStalls | None = None,
) -> None:
    """Apply one instruction to `state`, optionally recording what it violates.

    Every instruction is walked twice over a run: once while the dataflow is
    still converging, with reporting off, and once after it settles with
    reporting on. The state transitions are identical either way, which is what
    makes the reporting pass trustworthy.
    """
    instr = program.instructions[index]
    word = instr.word
    if word is None:
        return

    recording = report is not None and seen is not None

    control = word.control
    wait_mask = control["wait_mask"]
    write_barrier = control["write_barrier"]
    read_barrier = control["read_barrier"]
    # a zero stall is the safe encoding rather than zero cycles; see encoding.py
    raw_stall = control["stall"]
    yielded = raw_stall == STALL_YIELD
    stall = effective_stall(raw_stall)

    access = operand_access(instr.mnemonic, instr.operands)
    record = model.lookup(instr.opcode)
    if report is not None and record.note.startswith("opcode not in the model"):
        report.unknown_opcodes.add(instr.opcode)

    # a wait takes effect before the instruction reads its operands
    state.satisfy(wait_mask)
    # so everything below sees the state this instruction actually executes in

    # sorted only when reporting, where the order decides how findings read;
    # the convergence passes reach the same fixed point in any order
    uses = sorted(access.real_uses, key=str) if recording else access.real_uses
    for reg in uses:
        for rd in state.reaching(reg):
            producer = program.instructions[rd.index]
            if _exclusive(producer.text, instr.text):
                continue
            producer_record = model.lookup(producer.opcode)
            if report is not None:
                report.checked_pairs += 1

            # a wait covers the dependency whatever the latency, so it is checked
            # before the class is consulted
            if rd.yielded or (rd.barrier != NO_BARRIER and rd.satisfied):
                # a wait covers the bulk, not the residue: `DADD` is always
                # scoreboarded, never given less than 2, and wrong at 1
                _check_scoreboarded_minimum(
                    report, seen, program, observed, rd, reg, access.guard, index, instr, recording
                )
                continue

            # a variable-latency result the vendor is known to cover with
            # spacing is checked against that distance, not merely noted
            by_distance = producer_record.kind is LatencyClass.VARIABLE and (
                observed is not None
                and observed.covered_by_spacing(producer.opcode) >= MIN_OBSERVATIONS
            )

            if producer_record.kind is LatencyClass.FIXED or by_distance:
                required, source, grounded = _requirement(
                    producer.mnemonic,
                    instr.opcode,
                    producer_record,
                    observed,
                    guard=reg == access.guard,
                    bounded=not by_distance,
                )
                if rd.elapsed >= required or not recording:
                    continue
                # grounded in hardware or in what the vendor schedules, either of
                # which makes a shortfall an error rather than a suspicion
                severity = (
                    Severity.ERROR
                    if grounded or producer_record.confidence is Confidence.MEASURED
                    else Severity.WARNING
                )
                _add(
                    report,
                    seen,
                    Hazard(
                        kind=HazardKind.UNDERSTALLED,
                        severity=severity,
                        confidence=producer_record.confidence,
                        register=str(reg),
                        def_index=rd.index,
                        use_index=index,
                        def_text=producer.text,
                        use_text=instr.text,
                        required=required,
                        actual=rd.elapsed,
                        detail=source,
                    ),
                )

            elif producer_record.kind is LatencyClass.VARIABLE and recording:
                if rd.barrier == NO_BARRIER:
                    _add(
                        report,
                        seen,
                        Hazard(
                            kind=HazardKind.NO_BARRIER_SET,
                            severity=Severity.ERROR,
                            confidence=producer_record.confidence,
                            register=str(reg),
                            def_index=rd.index,
                            use_index=index,
                            def_text=producer.text,
                            use_text=instr.text,
                            detail=(
                                f"{producer.opcode} completes out of order but signals no "
                                "scoreboard, so nothing can wait for it"
                            ),
                        ),
                    )
                elif not rd.satisfied:
                    _add(
                        report,
                        seen,
                        Hazard(
                            kind=HazardKind.BARRIER_NOT_AWAITED,
                            severity=Severity.ERROR,
                            confidence=producer_record.confidence,
                            register=str(reg),
                            def_index=rd.index,
                            use_index=index,
                            def_text=producer.text,
                            use_text=instr.text,
                            detail=(
                                f"producer signals scoreboard {rd.barrier}, and on at least one "
                                f"path nothing waits on it before here (mask {wait_mask:#04x})"
                            ),
                        ),
                    )

    # write-after-read: overwriting an operand something else is still reading
    if recording:
        for sb, pending in state.pending_readers():
            if (wait_mask >> sb) & 1:
                continue
            for reg in access.real_defs:
                if reg not in pending.registers:
                    continue
                reader = program.instructions[pending.index]
                _add(
                    report,
                    seen,
                    Hazard(
                        kind=HazardKind.OVERWRITTEN_BEFORE_READ,
                        # a warning: elapsed cycles cover this as well as a wait
                        # does, and ptxas relies on the second (finding 13)
                        severity=Severity.WARNING,
                        confidence=record.confidence,
                        register=str(reg),
                        def_index=pending.index,
                        use_index=index,
                        def_text=reader.text,
                        use_text=instr.text,
                        detail=(
                            f"the earlier instruction signals read barrier {sb} because it has "
                            f"not consumed {reg} yet, and this write does not wait on it"
                        ),
                    ),
                )

    # a scoreboard signalled here also covers anything the same unit was still
    # owing, since a unit returns results in the order it was given work
    if write_barrier != NO_BARRIER:
        state.adopt(
            write_barrier,
            lambda other: program.instructions[other].opcode == instr.opcode,
        )

    for reg in access.real_defs:
        state.define(reg, index, write_barrier, conditional=access.guard is not None)

    if access.real_uses:
        state.begin_read(read_barrier, index, frozenset(access.real_uses))

    state.advance(stall, yielded=yielded)


def _check_anti_dependencies(program, block, report, seen, observed) -> None:
    """Overwriting a register something is still reading, without a barrier.

    The other direction of dependency (finding 23), and the half the scheduler
    charges for. Reported only where the vendor has been seen to leave a wider
    gap for this exact pairing enough times to trust the number: `ULEA` into
    `UMOV` is the only figure fault injection has pinned, and the vendor leaves
    one or two cycles for hundreds of other pairings, so a fixed constant here
    would produce 381 false alarms on `ptxas`'s own output.

    Within a block. A gap that spans an edge is a minimum over paths, which is
    the same reason the scoreboard residual is exempted there.
    """
    if report is None or seen is None or observed is None:
        return
    last_read: dict[RegRef, tuple[int, str]] = {}
    elapsed: dict[RegRef, int] = {}
    for index in range(block.start, block.end):
        instr = program.instructions[index]
        if instr.word is None:
            continue
        access = operand_access(instr.mnemonic, instr.operands)

        for reg in access.real_defs:
            previous = last_read.get(reg)
            if previous is None or previous[0] == index:
                continue
            reader_index, reader_mnemonic = previous
            reader = program.instructions[reader_index]
            # a read barrier covers this, and its wait is not a cycle count
            if reader.word is None or reader.word.field("read_barrier") != NO_BARRIER:
                continue
            evidence = observed.anti_requirement(reader_mnemonic, instr.opcode)
            if evidence is None:
                continue
            needed = anti_dependency_cycles(reader_mnemonic, instr.opcode, observed)
            if elapsed.get(reg, 0) >= needed:
                continue
            # the constant was measured for one pairing, so it grounds an error
            # only where this pairing's own evidence is what binds
            grounded = evidence.minimum <= ANTI_DEPENDENCY_CYCLES
            _add(
                report,
                seen,
                Hazard(
                    kind=HazardKind.OVERWRITTEN_BEFORE_READ,
                    severity=Severity.ERROR if grounded else Severity.WARNING,
                    confidence=Confidence.MEASURED if grounded else Confidence.ASSUMED,
                    register=str(reg),
                    def_index=reader_index,
                    use_index=index,
                    def_text=reader.text,
                    use_text=instr.text,
                    detail=(
                        f"{reg} is overwritten {elapsed.get(reg, 0)} cycles after it is read, "
                        f"and this pairing needs {needed}: ptxas never leaves it less than "
                        f"{evidence.minimum} across {evidence.observations} observations"
                    ),
                ),
            )

        stall = effective_stall(instr.word.field("stall"))
        for key in list(elapsed):
            elapsed[key] += stall
        for reg in access.real_defs:
            last_read.pop(reg, None)
            elapsed.pop(reg, None)
        for reg in access.real_uses:
            last_read[reg] = (index, instr.mnemonic)
            elapsed[reg] = stall


def _run_block(
    cfg: ControlFlowGraph,
    block_index: int,
    entry: FlowState,
    model: LatencyModel,
    report: VerificationReport | None,
    seen: set[tuple] | None,
    observed: ObservedStalls | None = None,
) -> FlowState:
    state = entry.copy()
    block = cfg.blocks[block_index]
    for index in range(block.start, block.end):
        _check_instruction(cfg.program, index, state, model, report, seen, observed)
    _check_anti_dependencies(cfg.program, block, report, seen, observed)
    return state


def verify_program(
    program: Program | list[Instruction],
    model: LatencyModel,
    observed: ObservedStalls | None = None,
) -> VerificationReport:
    """Verify a decoded kernel against a latency model.

    Accepts a `Program` for cross-block analysis, or a bare instruction list,
    which carries no labels and so degrades to per-block checking rather than
    guessing where branches go.

    `observed`, when supplied, refines the requirement per producer/consumer
    pairing rather than per producer. The requirement is genuinely a property of
    the pair: `IMAD` feeding another `IMAD` needs four cycles, while `IMAD`
    feeding an `IADD` is scheduled at three, because the consumer reads its
    operands a cycle later. Without that refinement a checker calibrated on the
    stricter pairing rejects real compiler output.
    """
    if not isinstance(program, Program):
        program = Program(instructions=list(program), labels={})

    cfg = build_cfg(program)
    report = VerificationReport(
        instructions=len(program.instructions),
        blocks=len(cfg.blocks),
        model_confidence=model.weakest_confidence,
        model_sku=model.sku,
        cross_block=any(b.successors for b in cfg.blocks),
        incomplete_graph=cfg.has_indirect_edges,
    )
    if not cfg.blocks:
        return report

    # ---- converge -------------------------------------------------------
    entries: list[FlowState] = [FlowState() for _ in cfg.blocks]
    worklist: list[int] = [0]
    iterations = 0

    while worklist and iterations < MAX_ITERATIONS:
        iterations += 1
        current = worklist.pop()
        exit_state = _run_block(cfg, current, entries[current], model, None, None, observed)
        # anything leaving a block is marked as having crossed one, which is what
        # lets a rule that only holds for a measurable distance say so
        outgoing = exit_state.crossing()
        for succ in cfg.blocks[current].successors:
            if entries[succ].merge(outgoing) and succ not in worklist:
                worklist.append(succ)

    # ---- report ---------------------------------------------------------
    # one pass over the settled states, so each finding is reported once
    seen: set[tuple] = set()
    for block in cfg.blocks:
        _run_block(cfg, block.index, entries[block.index], model, report, seen, observed)

    report.hazards.sort(key=lambda h: (h.use_index, h.def_index, h.register))
    return report


def split_blocks(instructions: list[Instruction]) -> list:
    """Basic blocks for a bare instruction list, without running the analysis."""
    return build_cfg(Program(instructions=list(instructions), labels={})).blocks


def _check_scoreboarded_minimum(
    report,
    seen,
    program,
    observed,
    rd,
    reg,
    guard,
    index: int,
    instr,
    recording: bool,
) -> None:
    """A waited-on scoreboard still leaves a gap the producer has to cover.

    The wait covers the long, variable part of the result and not the whole of
    it. `ptxas` scoreboards every `DADD` in the corpus and still never puts one
    less than 2 cycles from its consumer, and closing that to 1 while leaving
    the wait in place changes what the GPU computes. Treating a wait as a
    complete answer is what let basalt emit an fp64 kernel its own checker
    accepted and the hardware disagreed with.

    The quantity is the gap between producer and consumer, not the producer's
    own stall, because they are not the same: `FLO.U32` is scheduled 1 cycle
    from a consumer three instructions later and 2 from the one straight after
    it. Never reported against the safe stall encoding, which covers anything.

    Bounded by the measured residue, because the wait is what makes the pair
    safe and the gap beside it is spacing the compiler had work to fill:
    `LDG.E.64 => FFMA` mines at 461 cycles, and reporting anything short of that
    was 945 of the first 1,149 warnings the audit raised against shipped code.
    """
    if observed is None or report is None or not recording:
        return
    if rd.crossed or rd.index == index:
        # a minimum over paths, against evidence mined one block at a time: there
        # is nothing here to compare fairly
        return
    producer = program.instructions[rd.index]
    if producer.word is None or producer.word.field("stall") == STALL_YIELD:
        return
    consumer_key = ("@" if reg == guard else "") + instr.opcode
    evidence = observed.scoreboarded_minimum(producer.mnemonic, consumer_key)
    if evidence is None:
        return
    # the wait is what makes a scoreboarded pair safe, so the gap beside it
    # carries no requirement: only the measured residue grounds a claim here
    required = min(evidence.minimum, SCOREBOARD_RESIDUE_CYCLES)
    if rd.elapsed >= required:
        return
    _add(
        report,
        seen,
        Hazard(
            kind=HazardKind.UNDERSTALLED,
            severity=Severity.ERROR,
            confidence=Confidence.MEASURED,
            register=str(rd.register) if hasattr(rd, "register") else "",
            def_index=rd.index,
            use_index=index,
            def_text=f"{producer.mnemonic} {producer.operands}".strip(),
            use_text=f"{instr.mnemonic} {instr.operands}".strip(),
            required=required,
            actual=rd.elapsed,
            detail=(
                f"{producer.mnemonic} signals a scoreboard and is waited on, but a "
                f"scoreboard does not cover the whole result: the residue is measured at "
                f"{SCOREBOARD_RESIDUE_CYCLES} cycles, and the compiler was never seen to "
                f"schedule {evidence.producer} closer than {evidence.minimum} to "
                f"{evidence.consumer} across {evidence.observations} observations"
            ),
        ),
    )


def _requirement(
    producer: str,
    consumer: str,
    record: LatencyRecord,
    observed: ObservedStalls | None,
    *,
    guard: bool = False,
    bounded: bool = True,
) -> tuple[int, str, bool]:
    """How many cycles this pairing needs, where that came from, and how firmly.

    The third value says whether the number is grounded: measured on hardware or
    mined from what the vendor actually schedules, as opposed to an assumed
    producer latency. It decides whether falling short is an error or a warning,
    and it has to come from here because the strength of a requirement belongs to
    the requirement rather than to the producer's generic latency entry. Deciding
    it from the latter reported `IADD -> STG` at one cycle of the five the vendor
    never goes below as a warning, and the GPU computes a different answer there.

    A per-pair observation beats the producer's generic latency when one exists,
    because the requirement really does depend on both ends: a consumer that
    reads its operands later tolerates a shorter gap. Falling back to the
    producer figure keeps a pairing the compiler never emitted checkable, just
    more strictly.

    `bounded` says the producer's latency is a real number rather than the zero
    a variable-latency entry carries, so it can serve as the ceiling.

    When the value is the consumer's guard the pairing is a different one. A
    guard has to be resolved before the instruction issues at all, so it needs
    far more lead than the same predicate read as data, and it is looked up and
    reported under its own key.
    """
    key = ("@" if guard else "") + consumer
    if observed is not None:
        evidence = observed.requirement(producer, key)
        if evidence is not None:
            # a guard has a measured requirement, and a mined figure is an upper
            # bound, so evidence may lower it and never raise it
            if guard and evidence.minimum > GUARD_CYCLES:
                return (
                    GUARD_CYCLES,
                    f"a guard predicate needs {GUARD_CYCLES} cycles, measured; the vendor was "
                    f"seen no tighter than {evidence.minimum} but an upper bound cannot raise it",
                    True,
                )
            # after its own latency the result is written, so a mined figure
            # above it is spacing, and thin evidence is harmless once bounded
            ceiling = record.cycles if bounded else evidence.minimum
            return (
                min(evidence.minimum, ceiling),
                f"{evidence.producer} -> {evidence.consumer} is scheduled no tighter than "
                f"{evidence.minimum} cycles across {evidence.observations} observations",
                True,
            )
    if guard:
        return (
            GUARD_CYCLES,
            f"a guard predicate needs {GUARD_CYCLES} cycles, measured "
            f"(see docs/FINDINGS.md); {producer} is fixed latency",
            True,
        )
    return (
        record.cycles,
        f"{producer} latency is {record.confidence} ({record.note or 'no note'})",
        False,
    )
