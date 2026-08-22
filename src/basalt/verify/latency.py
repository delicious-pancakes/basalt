# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""How long each instruction takes, and how confident we are about it.

A latency checker is only as good as its latency model, so this module refuses
to pretend. Every entry carries provenance: whether the number was measured on
this machine's silicon, taken from published characterisation of a different
part, or assumed from architectural convention. The verifier reports findings
differently depending on which, because a hazard derived from an assumed number
is a lead and a hazard derived from a measured one is a bug.

The distinction that actually drives the analysis is not the number, though. It
is whether an instruction has a *fixed* latency the compiler must cover with
stall counts, or a *variable* latency it must cover with a scoreboard. Getting
that classification wrong produces either false alarms on every memory access or
silence on every arithmetic hazard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = [
    "DEFAULT_MODEL",
    "GUARD_CYCLES",
    "Confidence",
    "LatencyClass",
    "LatencyModel",
    "LatencyRecord",
]


class Confidence(StrEnum):
    """Where a latency number came from. Ordered weakest to strongest."""

    ASSUMED = "assumed"  # architectural convention, not verified anywhere
    PUBLISHED = "published"  # measured by someone else, on some other part
    MEASURED = "measured"  # measured by basalt, on a named SKU


class LatencyClass(StrEnum):
    """How the compiler is required to cover an instruction's latency."""

    FIXED = "fixed"  # covered by stall counts; no interlock, so errors are silent
    VARIABLE = "variable"  # covered by a scoreboard; the hardware does enforce the wait
    CONTROL = "control"  # branches and barriers, no register result to protect


@dataclass(frozen=True, slots=True)
class LatencyRecord:
    """One opcode's timing, with the evidence behind it."""

    cycles: int
    kind: LatencyClass
    confidence: Confidence
    note: str = ""
    source: str = ""

    @property
    def is_fixed(self) -> bool:
        return self.kind is LatencyClass.FIXED


# opcode prefixes by pipeline, longest-prefix on the bare opcode. ASSUMED
# throughout: stage 7 replaces these with numbers measured on silicon
# a guard is consumed at issue and needs 13 cycles where the same predicate read
# as data needs 5, so this is a property of issue rather than of the file
GUARD_CYCLES = 13

_ASSUMED: dict[str, tuple[int, LatencyClass, str]] = {
    # core integer and float ALU, the classic four-stage result bus
    "IADD": (4, LatencyClass.FIXED, "integer add pipeline"),
    "IADD3": (4, LatencyClass.FIXED, "three-input integer add"),
    "IMAD": (4, LatencyClass.FIXED, "integer multiply-add, also used for moves"),
    "IABS": (4, LatencyClass.FIXED, ""),
    "IMNMX": (4, LatencyClass.FIXED, ""),
    "ISETP": (4, LatencyClass.FIXED, "writes predicates"),
    "ISCADD": (4, LatencyClass.FIXED, ""),
    "LEA": (4, LatencyClass.FIXED, "address arithmetic"),
    "LOP3": (4, LatencyClass.FIXED, "three-input bitwise lookup"),
    "LOP": (4, LatencyClass.FIXED, ""),
    "SHF": (4, LatencyClass.FIXED, "funnel shift"),
    "PRMT": (4, LatencyClass.FIXED, "byte permute"),
    "SEL": (4, LatencyClass.FIXED, ""),
    "PLOP3": (4, LatencyClass.FIXED, "predicate lookup"),
    # timed at 18 and scoreboarded, but a scoreboard beside a long stall does not
    # make the stall redundant, so this stays fixed
    "POPC": (18, LatencyClass.VARIABLE, "population count, measured, scoreboard signalled"),
    "FLO": (4, LatencyClass.VARIABLE, "find leading one, scoreboard signalled"),
    # Scoreboarded in two of the three dependent instances in the corpus, and the
    # third is covered by a wait on a later instruction from the same unit rather
    # than by elapsed cycles. Left as fixed latency it produced a non-
    # deterministic `brev` on hardware from basalt's own schedule.
    "BREV": (4, LatencyClass.VARIABLE, "bit reverse, scoreboard signalled"),
    "MOV": (4, LatencyClass.FIXED, ""),
    "FADD": (4, LatencyClass.FIXED, ""),
    "FMUL": (4, LatencyClass.FIXED, ""),
    "FFMA": (4, LatencyClass.FIXED, ""),
    "FSETP": (4, LatencyClass.FIXED, ""),
    "FMNMX": (4, LatencyClass.FIXED, ""),
    "FSEL": (4, LatencyClass.FIXED, ""),
    "FCHK": (4, LatencyClass.FIXED, ""),
    "HADD2": (4, LatencyClass.FIXED, "packed half"),
    "HMUL2": (4, LatencyClass.FIXED, "packed half"),
    "HFMA2": (4, LatencyClass.FIXED, "packed half"),
    "HSETP2": (4, LatencyClass.FIXED, ""),
    "VIMNMX": (4, LatencyClass.FIXED, "packed integer min/max"),
    "VIADD": (4, LatencyClass.FIXED, ""),
    "VABSDIFF": (4, LatencyClass.FIXED, ""),
    "IDP": (4, LatencyClass.FIXED, "dot-product accumulate"),
    # the conversion pipe signals a scoreboard; calling it fixed latency made
    # basalt's own conversion schedules non-deterministic on hardware
    "I2F": (6, LatencyClass.VARIABLE, "conversion pipeline, scoreboard signalled"),
    "F2I": (6, LatencyClass.VARIABLE, "conversion pipeline, scoreboard signalled"),
    "F2F": (6, LatencyClass.VARIABLE, "conversion pipeline, scoreboard signalled"),
    # `I2FP` is a different instruction, not `I2F` with a suffix, and the
    # longest-prefix rule would otherwise hand it the conversion pipe's entry
    "I2FP": (6, LatencyClass.FIXED, "packed integer to float, never scoreboarded"),
    # `I2I` does not appear anywhere in the corpus, so there is no evidence for
    # its class either way and it keeps the conservative default rather than
    # being classed with the rest of the pipe on the strength of its name.
    "I2I": (6, LatencyClass.FIXED, "conversion pipeline, class unobserved"),
    "FRND": (6, LatencyClass.FIXED, ""),
    # fp64 completes out of order and is always scoreboarded; the wait carries
    # the dependency and `by_scoreboarded` carries what it still owes
    "DADD": (64, LatencyClass.VARIABLE, "fp64 add, measured, scoreboard signalled"),
    "DMUL": (64, LatencyClass.VARIABLE, "fp64 multiply, assumed equal to DADD"),
    "DFMA": (64, LatencyClass.VARIABLE, "fp64 fused multiply-add, measured"),
    "DSETP": (64, LatencyClass.VARIABLE, "fp64 compare, assumed equal to DADD"),
    # the special-function unit signals completion rather than running to a
    # fixed schedule
    "MUFU": (0, LatencyClass.VARIABLE, "special function unit, scoreboard signalled"),
    # tensor cores: fixed but long, and the exact figure differs per shape
    "HMMA": (16, LatencyClass.FIXED, "tensor core, shape dependent"),
    "IMMA": (16, LatencyClass.FIXED, "tensor core, shape dependent"),
    "QMMA": (16, LatencyClass.FIXED, "tensor core, low precision, shape dependent"),
    "OMMA": (16, LatencyClass.FIXED, "tensor core, block scaled, shape dependent"),
    "BMMA": (16, LatencyClass.FIXED, "tensor core, shape dependent"),
    # memory and anything else that completes out of order
    "LD": (0, LatencyClass.VARIABLE, ""),
    "LDG": (0, LatencyClass.VARIABLE, "global load"),
    "LDS": (0, LatencyClass.VARIABLE, "shared load"),
    "LDL": (0, LatencyClass.VARIABLE, "local load"),
    "LDC": (0, LatencyClass.VARIABLE, "constant load"),
    "LDCU": (0, LatencyClass.VARIABLE, "uniform constant load"),
    "LDSM": (0, LatencyClass.VARIABLE, "matrix load from shared"),
    # the matrix transpose unit signals completion the same way the matrix load
    # does, and is scheduled with a scoreboard in every instance the corpus has
    "MOVM": (0, LatencyClass.VARIABLE, "matrix transpose, scoreboard signalled"),
    "LDGSTS": (0, LatencyClass.VARIABLE, "async global to shared copy"),
    "ATOM": (0, LatencyClass.VARIABLE, ""),
    "ATOMG": (0, LatencyClass.VARIABLE, ""),
    "ATOMS": (0, LatencyClass.VARIABLE, ""),
    "RED": (0, LatencyClass.VARIABLE, "reduction, no register result"),
    "TEX": (0, LatencyClass.VARIABLE, ""),
    "TLD": (0, LatencyClass.VARIABLE, ""),
    "SULD": (0, LatencyClass.VARIABLE, ""),
    "S2R": (0, LatencyClass.VARIABLE, "special register read, scoreboard signalled"),
    "S2UR": (0, LatencyClass.VARIABLE, ""),
    "CS2R": (4, LatencyClass.FIXED, "the fast special-register path, unlike S2R"),
    "SHFL": (0, LatencyClass.VARIABLE, "warp shuffle"),
    # ptxas emits VOTEU with no scoreboard and reads UR the next instruction,
    # so it cannot be completing out of order
    "VOTEU": (2, LatencyClass.FIXED, "warp vote on the uniform datapath"),
    "UPOPC": (4, LatencyClass.FIXED, "uniform population count"),
    "UFLO": (4, LatencyClass.FIXED, ""),
    "CS2UR": (4, LatencyClass.FIXED, "uniform special-register read, fast path"),
    "MATCH": (0, LatencyClass.VARIABLE, ""),
    "R2UR": (0, LatencyClass.VARIABLE, ""),
    # uniform datapath mirrors the vector one
    "UIADD3": (4, LatencyClass.FIXED, "uniform integer add"),
    "UIMAD": (4, LatencyClass.FIXED, ""),
    "ULOP3": (4, LatencyClass.FIXED, ""),
    "ULEA": (4, LatencyClass.FIXED, ""),
    "UMOV": (4, LatencyClass.FIXED, ""),
    "USHF": (4, LatencyClass.FIXED, ""),
    "UISETP": (4, LatencyClass.FIXED, ""),
    "USEL": (4, LatencyClass.FIXED, ""),
    "UPRMT": (4, LatencyClass.FIXED, ""),
    # control flow and synchronisation produce no register result to protect
    "BRA": (0, LatencyClass.CONTROL, ""),
    "BRX": (0, LatencyClass.CONTROL, ""),
    "JMP": (0, LatencyClass.CONTROL, ""),
    "CALL": (0, LatencyClass.CONTROL, ""),
    "RET": (0, LatencyClass.CONTROL, ""),
    "EXIT": (0, LatencyClass.CONTROL, ""),
    "BSSY": (0, LatencyClass.CONTROL, ""),
    "BSYNC": (0, LatencyClass.CONTROL, ""),
    "BAR": (0, LatencyClass.CONTROL, ""),
    "WARPSYNC": (0, LatencyClass.CONTROL, ""),
    "MEMBAR": (0, LatencyClass.CONTROL, ""),
    "DEPBAR": (0, LatencyClass.CONTROL, ""),
    "NOP": (0, LatencyClass.CONTROL, ""),
    "YIELD": (0, LatencyClass.CONTROL, ""),
    "PMTRIG": (0, LatencyClass.CONTROL, ""),
    # stores have no destination register, so nothing downstream depends on them
    "ST": (0, LatencyClass.CONTROL, ""),
    "STG": (0, LatencyClass.CONTROL, ""),
    "STS": (0, LatencyClass.CONTROL, ""),
    "STL": (0, LatencyClass.CONTROL, ""),
    "STSM": (0, LatencyClass.CONTROL, ""),
}

# An opcode nobody has classified is treated as fixed with the longest common
# ALU latency. That direction is deliberate: it can produce a false alarm, which
# a human then investigates, rather than a false silence, which nobody ever sees.
UNKNOWN_DEFAULT = LatencyRecord(
    cycles=6,
    kind=LatencyClass.FIXED,
    confidence=Confidence.ASSUMED,
    note="opcode not in the model; assumed fixed and conservative",
)


@dataclass
class LatencyModel:
    """Latency lookup with provenance, optionally overlaid with measurements."""

    records: dict[str, LatencyRecord]
    sku: str = ""
    measured_from: str = ""

    @classmethod
    def assumed(cls) -> LatencyModel:
        return cls(
            records={
                op: LatencyRecord(
                    cycles=cycles,
                    kind=kind,
                    confidence=Confidence.ASSUMED,
                    note=note,
                    source="architectural convention",
                )
                for op, (cycles, kind, note) in _ASSUMED.items()
            }
        )

    def lookup(self, opcode: str) -> LatencyRecord:
        """Longest-prefix match on the bare opcode."""
        bare = opcode.split(".")[0].upper()
        if (exact := self.records.get(bare)) is not None:
            return exact
        best: LatencyRecord | None = None
        best_len = 0
        for key, rec in self.records.items():
            if bare.startswith(key) and len(key) > best_len:
                best, best_len = rec, len(key)
        return best or UNKNOWN_DEFAULT

    @property
    def weakest_confidence(self) -> Confidence:
        """The confidence a report should be qualified by."""
        order = [Confidence.ASSUMED, Confidence.PUBLISHED, Confidence.MEASURED]
        present = {r.confidence for r in self.records.values()}
        for c in order:
            if c in present:
                return c
        return Confidence.ASSUMED

    def overlay(self, path: Path) -> LatencyModel:
        """Apply measured latencies over the assumed baseline.

        The measurement file is produced by the latency harness on a real card.
        Only opcodes it names are replaced, so a partial measurement run is
        useful immediately instead of all-or-nothing.
        """
        raw = json.loads(path.read_text())
        merged = dict(self.records)
        for op, entry in raw.get("latencies", {}).items():
            merged[op.upper()] = LatencyRecord(
                cycles=int(entry["cycles"]),
                kind=LatencyClass(entry.get("kind", "fixed")),
                confidence=Confidence.MEASURED,
                note=entry.get("note", ""),
                source=raw.get("sku", str(path)),
            )
        return LatencyModel(
            records=merged,
            sku=raw.get("sku", ""),
            measured_from=str(path),
        )


DEFAULT_MODEL = LatencyModel.assumed()
