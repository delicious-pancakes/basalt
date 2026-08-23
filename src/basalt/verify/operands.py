# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Working out which registers an instruction reads and which it writes.

Every hazard question reduces to this one. `IADD R5, R5, 0x2a` writes R5 and
reads R5; `STG.E desc[UR4][R2.64], R7` writes nothing and reads UR4, R2, R3 and
R7. Getting the direction wrong in either place produces a verifier that is
confidently useless, so the rules here are explicit rather than clever.

Three things make this harder than splitting on commas:

*Width.* A `.64` or `.128` suffix means the named register is the base of a pair
or quad. `LDG.E.64 R2, ...` defines R2 and R3. Missing that is how a verifier
misses the exact hazard it exists to catch.

*Direction is not positional for every opcode.* SASS puts the destination first
almost everywhere, and stores, reductions, branches and barriers are the
"almost". Those are listed explicitly below rather than pattern-matched, because
a wrong guess here is silent.

*Predicates are registers too.* A guard `@!P0` is a read, and `ISETP` writes
predicates before it writes anything else.

*A guard is not an ordinary read.* It decides whether the instruction issues at
all, so it has to be resolved before issue rather than at operand-read time, and
it needs roughly two and a half times the lead. That is measured, not assumed:
see `docs/FINDINGS.md`. The guard is therefore recorded separately from the
other uses, so the checker and the scheduler can charge it correctly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache

__all__ = ["WIDTH_SUFFIXES", "Access", "RegKind", "RegRef", "operand_access"]


class RegKind(StrEnum):
    GENERAL = "R"  # R0..R254, R255 = RZ
    UNIFORM = "UR"  # UR0..UR62, UR63 = URZ
    PREDICATE = "P"  # P0..P6, P7 = PT
    UPREDICATE = "UP"
    BARRIER = "B"  # convergence barriers, not scoreboards


@dataclass(frozen=True, slots=True)
class RegRef:
    """One architectural register touched by an instruction."""

    kind: RegKind
    number: int

    @property
    def is_sink(self) -> bool:
        """RZ, URZ and PT read as constants and ignore writes.

        Treating them as real registers creates false dependencies between every
        instruction that happens to use the zero register, which is most of them.
        """
        return (
            (self.kind is RegKind.GENERAL and self.number == 255)
            or (self.kind is RegKind.UNIFORM and self.number == 63)
            or (self.kind in (RegKind.PREDICATE, RegKind.UPREDICATE) and self.number == 7)
        )

    def __str__(self) -> str:
        return f"{self.kind.value}{self.number}"


@dataclass
class Access:
    """Registers defined and used by one instruction."""

    defs: set[RegRef] = field(default_factory=set)
    uses: set[RegRef] = field(default_factory=set)
    # kept apart from the rest of `uses`: a guard is needed earlier than an
    # ordinary source and costs more to wait for
    guard: RegRef | None = None

    # built once: an Access is cached and read many times per fixed-point pass
    _real_defs: frozenset[RegRef] | None = field(default=None, repr=False, compare=False)
    _real_uses: frozenset[RegRef] | None = field(default=None, repr=False, compare=False)

    @property
    def real_defs(self) -> frozenset[RegRef]:
        if self._real_defs is None:
            self._real_defs = frozenset(r for r in self.defs if not r.is_sink)
        return self._real_defs

    @property
    def real_uses(self) -> frozenset[RegRef]:
        if self._real_uses is None:
            self._real_uses = frozenset(r for r in self.uses if not r.is_sink)
        return self._real_uses


# listed rather than inferred from the operand shape, which is wrong often
# enough to matter
_STORE_OPCODES = frozenset(
    {
        "ST",
        "STG",
        "STL",
        "STS",
        "STSM",
        "RED",
        "REDG",
        "SUST",
        "SURED",
        "STAS",
        "STLS",
    }
)

# Opcodes that define nothing at all.
_NO_DEF_OPCODES = frozenset(
    {
        "EXIT",
        "BRA",
        "BRX",
        "JMP",
        "JMX",
        "RET",
        "SYNC",
        "BSSY",
        "BSYNC",
        "NOP",
        "BAR",
        "MEMBAR",
        "DEPBAR",
        "ERRBAR",
        "PMTRIG",
        "CCTL",
        "CCTLL",
        "CCTLT",
        "PREFETCH",
        "PREFETCHG",
        "YIELD",
        "BPT",
        "RTT",
        "SAM",
        "RAM",
        "CALL",
        "BRK",
        "PBK",
        "CONT",
        "PCNT",
        "SSY",
        "PRET",
    }
)

# Atomics and loads-with-return define their first operand and also read memory,
# so they follow the default rule; they are named here only for documentation.

# Suffix -> how many consecutive registers the named base actually covers.
WIDTH_SUFFIXES: dict[str, int] = {
    "64": 2,
    "128": 4,
    "U8": 1,
    "S8": 1,
    "U16": 1,
    "S16": 1,
}

_REG_TOKEN = re.compile(r"\b(UR|R|UP|P|B)(\d+|Z|T)\b")
_PREDICATE_ONLY = re.compile(r"^!?~?U?P(\d+|T)$")
_GUARD = re.compile(r"^@(!)?(U?P)(\d+|T)\s+")
# a register written as R4.64 or R4.reuse; the dotted tail after a register
_REG_WITH_TAIL = re.compile(r"\b(UR|R)(\d+|Z)((?:\.[A-Za-z0-9_]+)*)")


def _mk(kind_text: str, num_text: str) -> RegRef | None:
    kind = {
        "R": RegKind.GENERAL,
        "UR": RegKind.UNIFORM,
        "P": RegKind.PREDICATE,
        "UP": RegKind.UPREDICATE,
        "B": RegKind.BARRIER,
    }.get(kind_text)
    if kind is None:
        return None
    if num_text == "Z":
        return RegRef(kind, 255 if kind is RegKind.GENERAL else 63)
    if num_text == "T":
        return RegRef(kind, 7)
    return RegRef(kind, int(num_text))


def _expand(base: RegRef, count: int) -> set[RegRef]:
    """A 64- or 128-bit access covers consecutive registers from the base.

    Only general and uniform registers are widened. A predicate is one bit
    whatever the instruction's data width, so widening `P0` on a 64-bit compare
    would invent a dependency on `P1` that does not exist.
    """
    if base.is_sink or count <= 1:
        return {base}
    if base.kind not in (RegKind.GENERAL, RegKind.UNIFORM):
        return {base}
    return {RegRef(base.kind, base.number + i) for i in range(count)}


# 64-bit operands occupy a register pair with no suffix saying so, and reading
# them as single registers loses half of every dependency
_PAIRED_OPCODES = frozenset({"DADD", "DMUL", "DFMA", "DSETP", "DMMA", "DMNMX"})

# the min/max family, whose register destination follows its predicate outputs
_PREDICATES_THEN_REGISTER = frozenset({"IMNMX", "FMNMX", "DMNMX", "HMNMX", "HMNMX2"})

# in `IMAD.WIDE dst, a, b, c` the destination and addend are register pairs and
# the factors are not; `.U32` describes the factors, so the suffix rule misreads it


def _in_descriptor(part: str, position: int) -> bool:
    """Does the token at `position` sit inside the `desc[...]` group?"""
    start = part.find("desc[")
    if start < 0:
        return False
    end = part.find("]", start)
    return start < position < end


def _width_from(mnemonic: str, tail: str) -> int:
    """How many registers a reference covers.

    The width can be attached to the register (`R2.64`), carried by the mnemonic
    (`LDG.E.64`), or implied by the opcode operating on 64-bit values at all.
    The per-register spelling wins because it is the most specific statement.
    """
    for part in tail.split("."):
        if part in WIDTH_SUFFIXES:
            return WIDTH_SUFFIXES[part]
    for part in mnemonic.split(".")[1:]:
        if part in WIDTH_SUFFIXES:
            return WIDTH_SUFFIXES[part]
    if mnemonic.split(".")[0].upper() in _PAIRED_OPCODES:
        return 2
    return 1


def _split_operands(text: str) -> list[str]:
    """Split on commas that are not inside brackets.

    `STG.E desc[UR4][R2.64], R7` has a comma-free bracket group, but plenty of
    forms carry commas inside `[...]`, so depth tracking is not optional.
    """
    out: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if last := "".join(current).strip():
        out.append(last)
    return out


# the dataflow fixed point asks the same instruction on every pass, and a
# library kernel is walked dozens of times before it settles
@lru_cache(maxsize=1 << 16)
def operand_access(mnemonic: str, operands: str) -> Access:
    """Determine the registers one instruction defines and uses.

    `mnemonic` carries the modifiers, e.g. `LDG.E.64`, because width lives there.

    Cached, so the result must stay read-only to its callers; every field on
    `Access` is a set built here and never mutated afterwards.
    """
    access = Access()
    opcode = mnemonic.split(".")[0].upper()

    body = operands
    if (guard := _GUARD.match(body)) is not None:
        if (p := _mk(guard.group(2), guard.group(3))) is not None:
            access.uses.add(p)
            if not p.is_sink:
                access.guard = p
        body = body[guard.end() :]

    parts = _split_operands(body)
    if not parts:
        return access

    defines_first = opcode not in _STORE_OPCODES and opcode not in _NO_DEF_OPCODES

    # instructions that write a predicate alongside a register print it first,
    # as in `ISETP.GE.AND P0, PT, R2, R3, PT` or `IADD R2, P1, R3, R4`
    leading_preds = 0
    for part in parts:
        if re.fullmatch(r"!?U?P(\d+|T)", part.strip()):
            leading_preds += 1
        else:
            break

    # one leading predicate then a register means both are written, as an atomic
    # or a shuffle does; two or more are the whole destination list
    wide = "WIDE" in mnemonic.split(".")[1:]

    def_slots = max(1, leading_preds)
    if leading_preds == 1 and len(parts) > 1 and _REG_WITH_TAIL.match(parts[1].strip()):
        def_slots = 2
    elif leading_preds == 0 and len(parts) > 1 and _PREDICATE_ONLY.match(parts[1].strip()):
        # a predicate straight after the register destination is a carry out, not
        # a source; a real predicate source appears last, as in `SEL ..., P1`
        def_slots = 2
    elif (
        leading_preds >= 2
        and opcode in _PREDICATES_THEN_REGISTER
        and len(parts) > leading_preds
        and _REG_WITH_TAIL.match(parts[leading_preds].strip())
    ):
        # these write a register behind two predicate outputs, where the compare
        # family has a source instead, so the shape cannot tell them apart
        def_slots = leading_preds + 1

    for idx, part in enumerate(parts):
        is_def_slot = defines_first and idx < def_slots
        # a bracketed operand is an address: every register inside it is read,
        # even when it sits in the destination slot
        addressed = "[" in part

        for m in _REG_WITH_TAIL.finditer(part):
            ref = _mk(m.group(1), m.group(2))
            if ref is None:
                continue
            if addressed:
                # a width on the mnemonic describes the data, not the address:
                # `STS.128 [R0], R8` writes four registers to one 32-bit address
                width = _width_from("", m.group(3))
                if _in_descriptor(part, m.start()):
                    width = max(width, 2)
            else:
                width = _width_from(mnemonic, m.group(3))
                if wide and (idx == 0 or idx == len(parts) - 1):
                    width = max(width, 2)
            regs = _expand(ref, width)
            if is_def_slot and not addressed:
                access.defs |= regs
            else:
                access.uses |= regs

        for m in _REG_TOKEN.finditer(part):
            if m.group(1) not in ("P", "UP", "B"):
                continue
            ref = _mk(m.group(1), m.group(2))
            if ref is None:
                continue
            if is_def_slot and not addressed:
                access.defs.add(ref)
            else:
                access.uses.add(ref)

    return access
