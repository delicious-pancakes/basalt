# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Driving nvdisasm, in both of the modes that matter.

nvdisasm gives us two independent oracles and basalt leans on both:

*cubin mode* takes a real ELF produced by ptxas and prints instructions with
their encodings. That is ground truth: every pair it emits is something the
vendor compiler actually generated, so the semantics are beyond dispute.

*raw mode* (`--binary SM120a`) takes a flat file of 16-byte words and decodes
whatever we hand it. That is the interesting one. It decodes encodings ptxas
will never emit, which is how basalt reaches instructions and operand forms
that have no PTX spelling.

Neither mode requires a GPU. Everything here runs on a machine with no NVIDIA
hardware at all, which is what makes the ISA database reproducible in CI.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .encoding import WORD_BYTES, Word
from .toolchain import Toolchain

__all__ = [
    "Instruction",
    "Program",
    "branch_target",
    "decode_word",
    "decode_words",
    "disassemble_cubin",
    "disassemble_program",
    "raw_arch",
]


def raw_arch(arch: str) -> str:
    """Convert a ptxas arch name to the spelling `nvdisasm --binary` expects.

    ptxas takes `sm_120a`; nvdisasm's raw mode takes `SM120a`. The suffix letter
    stays lower case, so this is not a plain upper(): `SM120A` is rejected.
    """
    name = arch.strip()
    if name.lower().startswith("sm_"):
        name = "SM" + name[3:]
    elif name.lower().startswith("sm"):
        name = "SM" + name[2:]
    return name


# /*0050*/   IADD R5, R5, 0x2a ;   /* 0x0000002a05057835 */
_LINE = re.compile(
    r"^\s*/\*(?P<offset>[0-9a-fA-F]+)\*/\s+"
    r"(?P<text>.*?)\s*"
    r"(?:/\*\s*(?P<lo>0x[0-9a-fA-F]+)\s*\*/)?\s*$"
)
# a continuation line carrying only the high half
_CONT = re.compile(r"^\s*/\*\s*(?P<hi>0x[0-9a-fA-F]+)\s*\*/\s*$")

# nvdisasm marks unknown encodings a few different ways depending on version;
# all of them are useful signal for the prober, so they are matched loosely.
_INVALID = re.compile(r"\b(?:\.INVALID|INVALID|\?\?\?|error)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Instruction:
    """One decoded instruction: where it sat, what it says, what it encodes to."""

    offset: int
    text: str
    word: Word | None

    @property
    def predicate(self) -> str:
        """The guard predicate, e.g. `@!P1`, or empty when unguarded.

        A guard is printed ahead of the mnemonic, so treating the first token as
        the opcode silently misreads every predicated instruction, which in real
        compiler output is a large minority of them.
        """
        head = self.text.split()
        return head[0] if head and head[0].startswith("@") else ""

    @property
    def _body(self) -> str:
        """The instruction with any guard removed."""
        text = self.text
        if self.predicate:
            text = text[len(self.predicate) :].lstrip()
        return text

    @property
    def mnemonic(self) -> str:
        """Opcode with its modifier suffixes, e.g. `IMAD.WIDE.U32`."""
        parts = self._body.split()
        return parts[0].rstrip(";") if parts else ""

    @property
    def opcode(self) -> str:
        """Bare opcode with modifiers stripped, e.g. `IMAD`."""
        return self.mnemonic.split(".")[0]

    @property
    def modifiers(self) -> tuple[str, ...]:
        return tuple(self.mnemonic.split(".")[1:])

    @property
    def operands(self) -> str:
        """Operand text, with the guard kept so a reader of it is still a read."""
        body = self._body.split(None, 1)
        tail = body[1].rstrip(" ;") if len(body) > 1 else ""
        return f"{self.predicate} {tail}".strip() if self.predicate else tail

    @property
    def is_valid(self) -> bool:
        return bool(self.text) and not _INVALID.search(self.text)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.offset:#06x}  {self.text}"


def _parse(listing: str) -> list[Instruction]:
    """Parse an nvdisasm listing into instructions.

    Handles the two-line encoding form (low half on the instruction line, high
    half on its own continuation line) as well as listings with no encodings
    at all, which is what raw mode produces without -hex.
    """
    out: list[Instruction] = []
    pending_lo: int | None = None

    for line in listing.splitlines():
        if not line.strip() or line.lstrip().startswith(("//", ".")):
            continue

        if (cont := _CONT.match(line)) is not None:
            if out and pending_lo is not None:
                hi = int(cont.group("hi"), 16)
                last = out[-1]
                out[-1] = Instruction(last.offset, last.text, Word.from_halves(pending_lo, hi))
                pending_lo = None
            continue

        if (m := _LINE.match(line)) is None:
            continue

        text = m.group("text").strip().rstrip(";").strip()
        if not text:
            continue

        lo = m.group("lo")
        pending_lo = int(lo, 16) if lo else None
        out.append(Instruction(int(m.group("offset"), 16), text, None))

    return out


@dataclass(frozen=True, slots=True)
class Program:
    """A disassembled kernel: its instructions and where its labels point.

    Labels are needed to build a control-flow graph. Without them the only
    honest analysis is per straight-line block, because a linear listing gives
    no way to know where a branch goes.
    """

    instructions: list[Instruction]
    labels: dict[str, int]  # label name -> index into `instructions`

    def __len__(self) -> int:
        return len(self.instructions)


# a label on its own line. not only the dotted form: taking that alone loses the
# call targets and the kernel's own entry
_LABEL = re.compile(r"^\s*([A-Za-z_.$][\w.$]*)\s*:\s*$")
# a branch target as nvdisasm prints it, e.g. `` `(.L_x_1) ``
_TARGET = re.compile(r"`\((?P<label>\.[A-Za-z_][\w.$]*)\)")


def branch_target(instr: Instruction) -> str | None:
    """The label a branch names, if it names one."""
    m = _TARGET.search(instr.text)
    return m.group("label") if m else None


def _parse_program(listing: str) -> Program:
    """Parse a listing into instructions plus a label index.

    Done in one pass alongside the instructions rather than by re-scanning,
    because a label's meaning is "the next instruction", and that is only
    unambiguous while walking the listing in order.
    """
    instructions: list[Instruction] = []
    labels: dict[str, int] = {}
    pending_labels: list[str] = []
    pending_lo: int | None = None

    for line in listing.splitlines():
        if not line.strip():
            continue

        if (lab := _LABEL.match(line)) is not None:
            pending_labels.append(lab.group(1))
            continue

        if line.lstrip().startswith(("//", ".")):
            continue

        if (cont := _CONT.match(line)) is not None:
            if instructions and pending_lo is not None:
                hi = int(cont.group("hi"), 16)
                last = instructions[-1]
                instructions[-1] = Instruction(
                    last.offset, last.text, Word.from_halves(pending_lo, hi)
                )
                pending_lo = None
            continue

        if (m := _LINE.match(line)) is None:
            continue

        text = m.group("text").strip().rstrip(";").strip()
        if not text:
            continue

        for name in pending_labels:
            labels[name] = len(instructions)
        pending_labels.clear()

        lo = m.group("lo")
        pending_lo = int(lo, 16) if lo else None
        instructions.append(Instruction(int(m.group("offset"), 16), text, None))

    return Program(instructions=instructions, labels=labels)


def disassemble_program(tc: Toolchain, cubin: Path, *, arch: str = "SM120a") -> Program:
    """Ground-truth oracle, keeping the labels a control-flow graph needs."""
    res = tc.run([str(tc.nvdisasm), "-c", "-hex", str(cubin)], check=False)
    if res.returncode != 0:
        return Program(instructions=[], labels={})

    program = _parse_program(res.stdout)
    keep = [i for i, instr in enumerate(program.instructions) if instr.word is not None]
    if len(keep) == len(program.instructions):
        return program

    # drop instructions with no encoding and renumber the labels to match
    remap = {old: new for new, old in enumerate(keep)}
    return Program(
        instructions=[program.instructions[i] for i in keep],
        labels={name: remap[idx] for name, idx in program.labels.items() if idx in remap},
    )


def disassemble_cubin(tc: Toolchain, cubin: Path, *, arch: str = "SM120a") -> list[Instruction]:
    """Ground-truth oracle: disassemble a real ptxas-produced ELF."""
    return disassemble_program(tc, cubin, arch=arch).instructions


# nvdisasm error   : Unrecognized operation for functional unit 'uC' at address 0x00000010
_ERR_ADDR = re.compile(r"at address\s+(0x[0-9a-fA-F]+)")


def _run_raw(tc: Toolchain, words: list[Word], arch: str) -> tuple[int, str, str]:
    blob = b"".join(w.to_bytes() for w in words)
    with tempfile.TemporaryDirectory(prefix="basalt-probe-") as tmp:
        path = Path(tmp) / "probe.bin"
        path.write_bytes(blob)
        res = tc.run([str(tc.nvdisasm), "-b", arch, str(path)], check=False, timeout=300.0)
    return res.returncode, res.stdout, res.stderr


def _decode_span(
    tc: Toolchain,
    words: list[Word],
    arch: str,
    start: int,
    end: int,
    out: list[Instruction | None],
) -> None:
    """Decode words[start:end], recording None for any word nvdisasm rejects.

    Raw mode aborts the whole file on the first illegal instruction and prints
    nothing, so a batch containing one bad word yields no output for the good
    ones either. It does name the offset it choked on, which is enough to split
    the span around the offender instead of falling back to one process per
    word. That difference is roughly two orders of magnitude on a full probe.
    """
    if start >= end:
        return

    rc, stdout, stderr = _run_raw(tc, words[start:end], arch)

    if rc == 0:
        for instr in _parse(stdout):
            idx = start + instr.offset // WORD_BYTES
            if start <= idx < end:
                out[idx] = Instruction(instr.offset, instr.text, words[idx])
        return

    if (m := _ERR_ADDR.search(stderr)) is not None:
        bad = start + int(m.group(1), 16) // WORD_BYTES
        if start <= bad < end:
            _decode_span(tc, words, arch, start, bad, out)  # clean prefix
            out[bad] = None  # the offender
            _decode_span(tc, words, arch, bad + 1, end, out)  # the rest
            return

    # no usable address in the diagnostic: bisect rather than give up, so a
    # single unparseable failure costs log(n) calls instead of the whole span
    if end - start == 1:
        out[start] = None
        return
    mid = (start + end) // 2
    _decode_span(tc, words, arch, start, mid, out)
    _decode_span(tc, words, arch, mid, end, out)


def decode_words(
    tc: Toolchain,
    words: Sequence[Word | int],
    *,
    arch: str = "SM120a",
) -> list[Instruction | None]:
    """Probe oracle: ask nvdisasm what an arbitrary sequence of words means.

    Returns a list positionally aligned with the input, holding None wherever
    nvdisasm refused to decode. Callers rely on that alignment: for the prober,
    "this word is not a legal instruction" is a measurement, not a failure.
    """
    normalised = [w if isinstance(w, Word) else Word(w) for w in words]
    out: list[Instruction | None] = [None] * len(normalised)
    _decode_span(tc, normalised, arch, 0, len(normalised), out)
    return out


def decode_word(tc: Toolchain, word: Word | int, *, arch: str = "SM120a") -> Instruction | None:
    """Single-word convenience wrapper. Prefer decode_words in bulk."""
    got = decode_words(tc, [word], arch=arch)
    return got[0] if got else None
