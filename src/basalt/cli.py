# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""basalt command line."""

from __future__ import annotations

import argparse
import sys
import textwrap

from . import __version__


def _doctor(args: argparse.Namespace) -> int:
    """Verify both oracles end to end and report what is usable."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from .disasm import decode_words, disassemble_cubin, raw_arch
    from .toolchain import ToolchainError, find_toolchain

    try:
        tc = find_toolchain(args.cuda_bin)
    except ToolchainError as exc:
        print(f"FAIL  toolchain: {exc}", file=sys.stderr)
        return 1

    print(f"ok    toolchain   {tc.version} in {tc.bin_dir}")

    probe_ptx = textwrap.dedent(
        """\
        .version 9.0
        .target sm_120a
        .address_size 64
        .visible .entry probe(.param .u64 p)
        {
            .reg .b32 %r<4>;
            .reg .b64 %rd<4>;
            ld.param.u64 %rd1, [p];
            cvta.to.global.u64 %rd2, %rd1;
            mov.u32 %r1, %tid.x;
            add.s32 %r2, %r1, 42;
            st.global.u32 [%rd2], %r2;
            ret;
        }
        """
    )

    with TemporaryDirectory(prefix="basalt-doctor-") as tmp:
        src, cubin = Path(tmp) / "probe.ptx", Path(tmp) / "probe.cubin"
        src.write_text(probe_ptx)

        res = tc.run([str(tc.ptxas), f"-arch={args.arch}", "-o", str(cubin), str(src)], check=False)
        if res.returncode != 0:
            print(f"FAIL  ptxas       {res.stderr.strip()}", file=sys.stderr)
            return 1
        print(f"ok    ptxas       assembled {args.arch}")

        insns = disassemble_cubin(tc, cubin)
        if not insns:
            print("FAIL  nvdisasm    cubin oracle produced no instructions", file=sys.stderr)
            return 1
        print(f"ok    cubin oracle{'':1s} {len(insns)} instructions with encodings")

        arch_raw = raw_arch(args.arch)
        back = decode_words(tc, [i.word for i in insns], arch=arch_raw)
        if not back:
            print(f"FAIL  probe oracle nvdisasm -b {arch_raw} rejected the batch", file=sys.stderr)
            return 1

        agree = sum(1 for a, b in zip(insns, back) if a.mnemonic == b.mnemonic)
        status = "ok   " if agree == len(insns) else "WARN "
        print(f"{status} probe oracle {agree}/{len(insns)} mnemonics round-tripped")

        if agree != len(insns):
            for a, b in zip(insns, back):
                if a.mnemonic != b.mnemonic:
                    print(f"        cubin={a.text!r}  probe={b.text!r}")
            return 1

    print("\nboth oracles healthy. no GPU required for anything above.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="basalt", description="a SASS assembler for sm_120")
    ap.add_argument("--version", action="version", version=f"basalt {__version__}")
    ap.add_argument("--cuda-bin", default=None, help="directory holding ptxas and nvdisasm")
    ap.add_argument("--arch", default="sm_120a", help="target architecture (default sm_120a)")

    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="verify the toolchain and both oracles")

    args = ap.parse_args(argv)
    return {"doctor": _doctor}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
