# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Checking that a program's control bits actually cover its data dependencies.

The rule the hardware does not enforce, stated precisely:

*Fixed-latency results.* An instruction at index `i` with latency `L` makes its
result available `L` cycles after it issues. The `stall` field of instruction
`k` is how many cycles pass before instruction `k+1` issues, so the elapsed time
between issuing `i` and issuing a consumer `j` is `sum(stall[i..j-1])`. The
consumer is safe only if that sum is at least `L`. Nothing checks this at run
time, which is why a violation is a wrong answer rather than a crash.

*Variable-latency results.* Memory and special-register reads finish whenever
they finish, so they signal a scoreboard through `write_barrier` and consumers
block on it through `wait_mask`. Here the hardware does enforce the wait, so the
failure mode is a consumer that never waits at all. Scoreboards are counters
rather than flags: several producers may signal the same one, and a single wait
covers every outstanding signal on it, for every instruction downstream.

*Operands still being read.* An instruction that consumes its sources late
signals a `read_barrier` meaning it has not finished reading them. Anything that
overwrites those registers must wait on that barrier, or the reader sees a value
written after it was scheduled to read.

Analysis is per basic block. Tracking a definition across a branch needs a real
control-flow graph, and inventing one from a linear listing produces confident
nonsense, so blocks end at control flow and definitions do not cross.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ..disasm import Instruction
from ..encoding import NO_BARRIER
from .latency import Confidence, LatencyClass, LatencyModel, LatencyRecord
from .operands import RegRef, operand_access

__all__ = [
    "BasicBlock",
    "Hazard",
    "HazardKind",
    "Severity",
    "VerificationReport",
    "split_blocks",
    "verify_program",
]

# Instructions that end a basic block. Anything after one of these may be
# reached from elsewhere, so a definition before it cannot be assumed live.
_BLOCK_ENDERS = frozenset(
    {
        "BRA",
        "BRX",
        "JMP",
        "JMX",
        "CALL",
        "RET",
        "EXIT",
        "BSSY",
        "BSYNC",
        "BAR",
        "WARPSYNC",
        "SYNC",
        "BRK",
        "CONT",
        "PBK",
        "PCNT",
        "SSY",
        "PRET",
        "RTT",
        "BPT",
    }
)


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

    def describe(self) -> str:
        where = f"#{self.def_index} -> #{self.use_index}"
        if self.kind is HazardKind.UNDERSTALLED:
            gap = f"{self.actual} of {self.required} cycles"
            return f"[{self.severity}] {where} {self.register}: {gap}. {self.detail}".rstrip()
        return f"[{self.severity}] {where} {self.register}: {self.detail}"


@dataclass
class BasicBlock:
    """A straight-line run of instructions with their original indices."""

    start: int
    instructions: list[Instruction] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.instructions)


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

    @property
    def ok(self) -> bool:
        return not any(h.severity is Severity.ERROR for h in self.hazards)

    def by_severity(self, severity: Severity) -> list[Hazard]:
        return [h for h in self.hazards if h.severity is severity]

    def summary(self) -> str:
        errors = len(self.by_severity(Severity.ERROR))
        warnings = len(self.by_severity(Severity.WARNING))
        verdict = "clean" if not self.hazards else f"{errors} errors, {warnings} warnings"
        return (
            f"{self.instructions} instructions in {self.blocks} blocks, "
            f"{self.checked_pairs} dependencies checked: {verdict}"
        )


def split_blocks(instructions: list[Instruction]) -> list[BasicBlock]:
    """Cut the listing into straight-line blocks at control flow."""
    blocks: list[BasicBlock] = []
    current = BasicBlock(start=0)

    for idx, instr in enumerate(instructions):
        current.instructions.append(instr)
        if instr.opcode.upper() in _BLOCK_ENDERS:
            blocks.append(current)
            current = BasicBlock(start=idx + 1)

    if current.instructions:
        blocks.append(current)
    return blocks


@dataclass(slots=True)
class _PendingRead:
    """An instruction that has signalled it has not consumed its sources yet."""

    index: int
    instr: Instruction
    registers: set[RegRef]


@dataclass(slots=True)
class _Def:
    """The last writer of a register inside the block under analysis."""

    index: int
    instr: Instruction
    record: LatencyRecord
    barrier: int  # write_barrier it signalled, or NO_BARRIER
    satisfied: bool = False  # some instruction has since waited on that barrier


def _stalls_between(block: BasicBlock, lo: int, hi: int) -> int:
    """Cycles elapsed between issuing block[lo] and issuing block[hi].

    The stall on an instruction delays the *next* issue, so the span runs from
    the definition's own stall through the instruction before the consumer.
    """
    total = 0
    for i in range(lo, hi):
        word = block.instructions[i].word
        if word is not None:
            total += word.field("stall")
    return total


def _verify_block(
    block: BasicBlock,
    model: LatencyModel,
    report: VerificationReport,
) -> None:
    defs: dict[RegRef, _Def] = {}
    # scoreboard -> definitions signalled on it that nothing has awaited yet.
    # A scoreboard is a counter rather than a flag: several producers may signal
    # the same one, and a single wait covers every outstanding signal on it.
    pending: dict[int, list[_Def]] = {}
    # read barrier -> instructions still consuming the registers they name
    reads_pending: dict[int, list[_PendingRead]] = {}

    for local, instr in enumerate(block.instructions):
        word = instr.word
        if word is None:
            continue

        control = word.control
        wait_mask = control["wait_mask"]
        write_barrier = control["write_barrier"]
        read_barrier = control["read_barrier"]

        access = operand_access(instr.mnemonic, instr.operands)
        record = model.lookup(instr.opcode)
        if record.note.startswith("opcode not in the model"):
            report.unknown_opcodes.add(instr.opcode)

        # ---- wait first: an instruction's own wait_mask takes effect before
        # it reads its operands, and satisfies every producer still outstanding
        # on those scoreboards, not only the one this instruction happens to
        # consume. Later instructions inherit that: once anything has waited,
        # the data is available to everyone downstream.
        for sb in list(pending):
            if (wait_mask >> sb) & 1:
                for waiting in pending.pop(sb):
                    waiting.satisfied = True

        # ---- consume: check every read against its producer ----------------
        for reg in sorted(access.real_uses, key=str):
            producer = defs.get(reg)
            if producer is None:
                continue
            report.checked_pairs += 1

            if producer.record.kind is LatencyClass.FIXED:
                elapsed = _stalls_between(block, producer.index, local)
                if elapsed < producer.record.cycles:
                    severity = (
                        Severity.ERROR
                        if producer.record.confidence is Confidence.MEASURED
                        else Severity.WARNING
                    )
                    report.hazards.append(
                        Hazard(
                            kind=HazardKind.UNDERSTALLED,
                            severity=severity,
                            confidence=producer.record.confidence,
                            register=str(reg),
                            def_index=block.start + producer.index,
                            use_index=block.start + local,
                            def_text=producer.instr.text,
                            use_text=instr.text,
                            required=producer.record.cycles,
                            actual=elapsed,
                            detail=(
                                f"{producer.instr.opcode} latency is {producer.record.confidence}"
                                f" ({producer.record.note or 'no note'})"
                            ),
                        )
                    )
            else:
                if producer.barrier == NO_BARRIER:
                    report.hazards.append(
                        Hazard(
                            kind=HazardKind.NO_BARRIER_SET,
                            severity=Severity.ERROR,
                            confidence=producer.record.confidence,
                            register=str(reg),
                            def_index=block.start + producer.index,
                            use_index=block.start + local,
                            def_text=producer.instr.text,
                            use_text=instr.text,
                            detail=(
                                f"{producer.instr.opcode} completes out of order but signals no "
                                "scoreboard, so the consumer cannot wait for it"
                            ),
                        )
                    )
                elif not producer.satisfied:
                    report.hazards.append(
                        Hazard(
                            kind=HazardKind.BARRIER_NOT_AWAITED,
                            severity=Severity.ERROR,
                            confidence=producer.record.confidence,
                            register=str(reg),
                            def_index=block.start + producer.index,
                            use_index=block.start + local,
                            def_text=producer.instr.text,
                            use_text=instr.text,
                            detail=(
                                f"producer signals scoreboard {producer.barrier}, but nothing "
                                f"between the two waits on it and this consumer's mask is "
                                f"{wait_mask:#04x}"
                            ),
                        )
                    )

        # ---- write-after-read: overwriting an operand still being read -----
        # An instruction that consumes its sources late signals a read barrier
        # meaning "I have not finished reading yet". Anything that overwrites
        # those sources has to wait on it, or the reader sees the new value.
        for reg in access.real_defs:
            for sb, readers in list(reads_pending.items()):
                for reader in readers:
                    if reg in reader.registers and not (wait_mask >> sb) & 1:
                        report.hazards.append(
                            Hazard(
                                kind=HazardKind.OVERWRITTEN_BEFORE_READ,
                                severity=Severity.ERROR,
                                confidence=record.confidence,
                                register=str(reg),
                                def_index=block.start + reader.index,
                                use_index=block.start + local,
                                def_text=reader.instr.text,
                                use_text=instr.text,
                                detail=(
                                    f"the earlier instruction signals read barrier {sb} because "
                                    f"it has not consumed {reg} yet, and this write does not "
                                    f"wait on it (mask {wait_mask:#04x})"
                                ),
                            )
                        )

        for sb in list(reads_pending):
            if (wait_mask >> sb) & 1:
                del reads_pending[sb]

        # ---- define: record this instruction's results ---------------------
        this_def = _Def(index=local, instr=instr, record=record, barrier=write_barrier)

        # only out-of-order results need a scoreboard; a fixed-latency
        # instruction may still signal one, which is legal and simply redundant
        if write_barrier != NO_BARRIER and record.kind is LatencyClass.VARIABLE:
            pending.setdefault(write_barrier, []).append(this_def)

        if read_barrier != NO_BARRIER and access.real_uses:
            reads_pending.setdefault(read_barrier, []).append(
                _PendingRead(index=local, instr=instr, registers=set(access.real_uses))
            )

        for reg in access.real_defs:
            defs[reg] = this_def


def verify_program(
    instructions: list[Instruction],
    model: LatencyModel,
) -> VerificationReport:
    """Verify a decoded instruction stream against a latency model."""
    blocks = split_blocks(instructions)
    report = VerificationReport(
        instructions=len(instructions),
        blocks=len(blocks),
        model_confidence=model.weakest_confidence,
        model_sku=model.sku,
    )
    for block in blocks:
        _verify_block(block, model, report)

    report.hazards.sort(key=lambda h: (h.use_index, h.def_index, h.register))
    return report
