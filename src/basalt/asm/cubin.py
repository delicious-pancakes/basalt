# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Reading and rewriting instruction words inside a cubin.

A cubin is an ELF64 object. The instruction stream for a kernel lives in a
section named `.text.<kernel>`, sixteen bytes per instruction, so locating a
kernel and rewriting one of its words is ordinary ELF work rather than anything
exotic.

This exists for two reasons and both are about being able to check basalt itself:

*Negative controls.* A verifier that reports "clean" on everything is
indistinguishable from a verifier that does nothing. Taking real compiler output,
shortening one stall count, and confirming the tool flags exactly that
instruction is the test that separates the two.

*Round-trip evidence.* Reading a section, decoding it, re-encoding it and getting
the same bytes back is the cheapest available proof that the encoding model is
faithful.

Only ELF64 little-endian is handled, which is all NVIDIA emits.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from ..encoding import WORD_BYTES, Word

__all__ = ["Cubin", "CubinError", "Section"]


class CubinError(RuntimeError):
    """Raised when a file is not a cubin we can work with."""


_ELFCLASS64 = 2
_ELFDATA2LSB = 1


@dataclass(frozen=True, slots=True)
class Section:
    """One ELF section header, with the fields we actually use."""

    name: str
    sh_type: int
    flags: int
    offset: int
    size: int
    index: int

    @property
    def is_text(self) -> bool:
        return self.name.startswith(".text.")

    @property
    def kernel(self) -> str:
        return self.name[len(".text.") :] if self.is_text else ""

    @property
    def instruction_count(self) -> int:
        return self.size // WORD_BYTES


class Cubin:
    """An in-memory cubin whose instruction words can be read and replaced."""

    def __init__(self, data: bytes) -> None:
        self._data = bytearray(data)
        self._sections: list[Section] = []
        self._parse()

    # ---- construction --------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> Cubin:
        return cls(Path(path).read_bytes())

    def save(self, path: Path) -> None:
        Path(path).write_bytes(bytes(self._data))

    @property
    def data(self) -> bytes:
        return bytes(self._data)

    # ---- ELF parsing ---------------------------------------------------

    def _parse(self) -> None:
        d = self._data
        if len(d) < 64 or d[:4] != b"\x7fELF":
            raise CubinError("not an ELF file")
        if d[4] != _ELFCLASS64:
            raise CubinError("only ELF64 is supported; this file is 32-bit")
        if d[5] != _ELFDATA2LSB:
            raise CubinError("only little-endian ELF is supported")

        # e_shoff, e_shentsize, e_shnum, e_shstrndx from the ELF64 header
        (e_shoff,) = struct.unpack_from("<Q", d, 0x28)
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", d, 0x3A)
        if e_shoff == 0 or e_shnum == 0:
            raise CubinError("no section headers")

        def header(i: int) -> tuple[int, int, int, int, int]:
            base = e_shoff + i * e_shentsize
            sh_name, sh_type, sh_flags = struct.unpack_from("<IIQ", d, base)
            sh_offset, sh_size = struct.unpack_from("<QQ", d, base + 0x18)
            return sh_name, sh_type, sh_flags, sh_offset, sh_size

        _, _, _, strtab_off, strtab_size = header(e_shstrndx)
        strtab = bytes(d[strtab_off : strtab_off + strtab_size])

        def name_at(off: int) -> str:
            end = strtab.find(b"\0", off)
            return strtab[off : end if end != -1 else None].decode("utf-8", "replace")

        for i in range(e_shnum):
            sh_name, sh_type, sh_flags, sh_offset, sh_size = header(i)
            self._sections.append(
                Section(
                    name=name_at(sh_name),
                    sh_type=sh_type,
                    flags=sh_flags,
                    offset=sh_offset,
                    size=sh_size,
                    index=i,
                )
            )

    # ---- queries -------------------------------------------------------

    @property
    def sections(self) -> list[Section]:
        return list(self._sections)

    @property
    def text_sections(self) -> list[Section]:
        return [s for s in self._sections if s.is_text and s.size >= WORD_BYTES]

    @property
    def kernels(self) -> list[str]:
        return [s.kernel for s in self.text_sections]

    def section(self, name: str) -> Section:
        for s in self._sections:
            if s.name == name:
                return s
        raise CubinError(f"no section named {name!r}")

    def text_for(self, kernel: str | None = None) -> Section:
        """The text section for a kernel, or the only one if unambiguous."""
        texts = self.text_sections
        if not texts:
            raise CubinError("no .text.* sections; is this a cubin?")
        if kernel is None:
            if len(texts) > 1:
                names = ", ".join(t.kernel for t in texts)
                raise CubinError(f"cubin holds several kernels ({names}); name one")
            return texts[0]
        for t in texts:
            if t.kernel == kernel:
                return t
        raise CubinError(f"no kernel named {kernel!r}")

    # ---- instruction access --------------------------------------------

    def words(self, kernel: str | None = None) -> list[Word]:
        sec = self.text_for(kernel)
        return [
            Word.from_bytes(bytes(self._data[o : o + WORD_BYTES]))
            for o in range(sec.offset, sec.offset + sec.size, WORD_BYTES)
        ]

    def read_word(self, index: int, kernel: str | None = None) -> Word:
        sec = self.text_for(kernel)
        if not 0 <= index < sec.instruction_count:
            raise IndexError(f"instruction {index} out of range for {sec.name}")
        base = sec.offset + index * WORD_BYTES
        return Word.from_bytes(bytes(self._data[base : base + WORD_BYTES]))

    def write_word(self, index: int, word: Word, kernel: str | None = None) -> None:
        """Replace one instruction in place.

        The section size does not change, so every offset, relocation and
        header in the file stays valid. That is the whole reason patching a
        word is safe while inserting one is not.
        """
        sec = self.text_for(kernel)
        if not 0 <= index < sec.instruction_count:
            raise IndexError(f"instruction {index} out of range for {sec.name}")
        base = sec.offset + index * WORD_BYTES
        self._data[base : base + WORD_BYTES] = word.to_bytes()

    def patch_control(
        self,
        index: int,
        field: str,
        value: int,
        kernel: str | None = None,
    ) -> Word:
        """Set one control field on one instruction, returning the new word."""
        patched = self.read_word(index, kernel).with_field(field, value)
        self.write_word(index, patched, kernel)
        return patched
