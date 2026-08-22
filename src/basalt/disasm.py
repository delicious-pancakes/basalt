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
from dataclasses import dataclass
from pathlib import Path

from .encoding import Word
from .toolchain import Toolchain

__all__ = ["Instruction", "disassemble_cubin", "decode_words", "decode_word", "raw_arch"]


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
    def mnemonic(self) -> str:
        """Opcode with its modifier suffixes, e.g. `IMAD.WIDE.U32`."""
        head = self.text.split()[0] if self.text.split() else ""
        return head.rstrip(";")

    @property
    def opcode(self) -> str:
        """Bare opcode with modifiers stripped, e.g. `IMAD`."""
        return self.mnemonic.split(".")[0]

    @property
    def modifiers(self) -> tuple[str, ...]:
        return tuple(self.mnemonic.split(".")[1:])

    @property
    def operands(self) -> str:
        body = self.text.split(None, 1)
        return body[1].rstrip(" ;") if len(body) > 1 else ""

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


def disassemble_cubin(tc: Toolchain, cubin: Path, *, arch: str = "SM120a") -> list[Instruction]:
    """Ground-truth oracle: disassemble a real ptxas-produced ELF."""
    res = tc.run([str(tc.nvdisasm), "-c", "-hex", str(cubin)], check=False)
    if res.returncode != 0:
        return []
    return [i for i in _parse(res.stdout) if i.word is not None]


def decode_words(
    tc: Toolchain,
    words: list[Word] | list[int],
    *,
    arch: str = "SM120a",
) -> list[Instruction]:
    """Probe oracle: ask nvdisasm what an arbitrary sequence of words means.

    Words are decoded in one batch because process startup dominates the cost
    by two orders of magnitude; the prober routinely pushes tens of thousands
    of candidates through a single call.
    """
    normalised = [w if isinstance(w, Word) else Word(w) for w in words]
    blob = b"".join(w.to_bytes() for w in normalised)

    with tempfile.TemporaryDirectory(prefix="basalt-probe-") as tmp:
        path = Path(tmp) / "probe.bin"
        path.write_bytes(blob)
        res = tc.run(
            [str(tc.nvdisasm), "-b", arch, str(path)],
            check=False,
            timeout=300.0,
        )

    if res.returncode != 0:
        # a whole-batch failure tells us nothing about which word caused it;
        # callers that care re-run the batch one word at a time.
        return []

    parsed = _parse(res.stdout)

    # raw mode emits one line per input word in order, so we can re-attach the
    # encodings we supplied rather than asking nvdisasm to print them back.
    out: list[Instruction] = []
    for idx, instr in enumerate(parsed[: len(normalised)]):
        out.append(Instruction(instr.offset, instr.text, normalised[idx]))
    return out


def decode_word(tc: Toolchain, word: Word | int, *, arch: str = "SM120a") -> Instruction | None:
    """Single-word convenience wrapper. Prefer decode_words in bulk."""
    got = decode_words(tc, [word], arch=arch)
    return got[0] if got else None
