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

from .encoding import WORD_BYTES, Word
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
            _decode_span(tc, words, arch, start, bad, out)      # clean prefix
            out[bad] = None                                     # the offender
            _decode_span(tc, words, arch, bad + 1, end, out)    # the rest
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
    words: list[Word] | list[int],
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
