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

        agree = sum(
            1
            for a, b in zip(insns, back, strict=True)
            if b is not None and a.mnemonic == b.mnemonic
        )
        status = "ok   " if agree == len(insns) else "WARN "
        print(f"{status} probe oracle {agree}/{len(insns)} mnemonics round-tripped")

        if agree != len(insns):
            for a, b in zip(insns, back, strict=True):
                if b is None or a.mnemonic != b.mnemonic:
                    print(f"        cubin={a.text!r}  probe={b.text if b else None!r}")
            return 1

    print("\nboth oracles healthy. no GPU required for anything above.")
    return 0


DEFAULT_DB = "data/isa/sm_120a.json"
DEFAULT_OBSERVED = "data/latency/observed-stalls-sm120a.json"


def _build_isa(args: argparse.Namespace) -> int:
    """Run harvest and probe, then write the database."""
    from pathlib import Path

    from .isa.build import build_database
    from .toolchain import ToolchainError, find_toolchain

    try:
        tc = find_toolchain(args.cuda_bin)
    except ToolchainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    db, _ = build_database(
        tc,
        arch=args.arch,
        include_tensor=not args.no_tensor,
        harvest_out=Path(args.harvest_out) if args.harvest_out else None,
    )
    out = Path(args.out)
    db.write(out)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def _isa(args: argparse.Namespace) -> int:
    """Query the database."""
    from pathlib import Path

    from .isa.database import IsaDatabase

    path = Path(args.db)
    if not path.exists():
        print(f"error: {path} not found; run `basalt build-isa` first", file=sys.stderr)
        return 1

    db = IsaDatabase.read(path)

    if args.stats:
        print(f"arch {db.arch}  built with ptxas {db.cuda_version}  {db.generated_utc}")
        for k, v in db.coverage().items():
            print(f"  {k.replace('_', ' '):<24} {v}")
        return 0

    if args.opcode:
        forms = db.by_opcode(args.opcode.upper())
        if not forms:
            print(f"no forms for opcode {args.opcode.upper()}", file=sys.stderr)
            return 1
        for f in forms:
            print(f.describe())
        return 0

    if args.mnemonic:
        form = db.get(args.mnemonic.upper())
        if form is None:
            print(f"unknown form {args.mnemonic.upper()}", file=sys.stderr)
            return 1
        print(f"{form.mnemonic}")
        print(f"  example    {form.operand_text}")
        print(f"  encoding   {form.encoding}")
        print(f"  from       {form.source_label} ({form.source_family})")
        for o in form.operands:
            print(f"  operand[{o.slot}] {o.width:3d} bits at {o.describe()}")
            if o.example_before:
                print(f"              {o.example_before}  ->  {o.example_after}")
        print(f"  opcode     {len(form.opcode_bits)} bits")
        print(f"  inert      {len(form.inert_bits)} bits")
        return 0

    for name in sorted(db.forms):
        print(name)
    return 0


def _probe_stalls(args: argparse.Namespace) -> int:
    """Determine the required stall for each opcode by fault injection."""
    from .gpu.driver import Device, cuda_available
    from .gpu.inject import probe_required_stall
    from .gpu.latency import _SPECS
    from .toolchain import ToolchainError, find_toolchain

    try:
        tc = find_toolchain(args.cuda_bin)
    except ToolchainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not cuda_available():
        print("error: no CUDA device; this command needs real silicon", file=sys.stderr)
        return 1

    print("sweeping the stall on a dependent pair and comparing against a reference")
    with Device(args.device) as dev:
        print(f"on {dev.info.describe()}")
        print()
        for spec in _SPECS:
            if args.opcode and spec.opcode.upper() != args.opcode.upper():
                continue
            result = probe_required_stall(
                tc, dev, spec, arch=args.arch, links=args.links, repeats=args.repeats
            )
            print(f"  {result.describe()}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    """Check a cubin's control bits against the latency model."""
    from pathlib import Path

    from .disasm import disassemble_program
    from .toolchain import ToolchainError, find_toolchain
    from .verify.hazards import Severity, verify_program
    from .verify.latency import DEFAULT_MODEL, Confidence, LatencyModel
    from .verify.observed import ObservedStalls

    try:
        tc = find_toolchain(args.cuda_bin)
    except ToolchainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    target = Path(args.cubin)
    if not target.is_file():
        print(f"error: {target} not found", file=sys.stderr)
        return 1

    model = DEFAULT_MODEL
    if args.latencies:
        model = LatencyModel.assumed().overlay(Path(args.latencies))

    observed = None
    if args.observed:
        observed = ObservedStalls.read(Path(args.observed))
    elif (default := Path(DEFAULT_OBSERVED)).is_file():
        observed = ObservedStalls.read(default)

    program = disassemble_program(tc, target)
    if not program.instructions:
        print(f"error: nothing disassembled from {target}", file=sys.stderr)
        return 1

    report = verify_program(program, model, observed=observed)

    print(f"{target}")
    print(f"  {report.summary()}")
    if observed is not None:
        print(f"  pair data: {len(observed.by_pair)} pairs mined from {observed.kernels} kernels")
    if model.sku:
        print(f"  latency model: measured on {model.sku}")
    else:
        print(f"  latency model: {report.model_confidence}, not measured on silicon")
    if report.incomplete_graph:
        print(
            "  note: some branch destinations are computed rather than named, so paths\n"
            "        through them were not analysed"
        )
    if report.unknown_opcodes:
        names = ", ".join(sorted(report.unknown_opcodes)[:8])
        print(f"  opcodes not in the model: {len(report.unknown_opcodes)} ({names})")

    shown = [h for h in report.hazards if args.all or h.severity is not Severity.INFO]
    if shown:
        print()
        for h in shown[: args.limit]:
            print(f"  {h.describe()}")
            print(f"      def  {h.def_text}")
            print(f"      use  {h.use_text}")
        if len(shown) > args.limit:
            print(f"  ... and {len(shown) - args.limit} more (raise --limit to see them)")

    if report.model_confidence is Confidence.ASSUMED and report.hazards:
        print(
            "\nnote: the latency model is assumed rather than measured, so these are leads\n"
            "      rather than confirmed hazards. measure the model first."
        )

    return 0 if report.ok or not args.strict else 2


def _measure(args: argparse.Namespace) -> int:
    """Measure instruction latency on a real device."""
    from pathlib import Path

    from .gpu.driver import cuda_available
    from .gpu.latency import measure_all
    from .toolchain import ToolchainError, find_toolchain

    try:
        tc = find_toolchain(args.cuda_bin)
    except ToolchainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not cuda_available():
        print(
            "error: no CUDA device. this is the one basalt command that needs real\n"
            "       silicon; everything else runs without a GPU.",
            file=sys.stderr,
        )
        return 1

    run = measure_all(
        tc,
        arch=args.arch,
        ordinal=args.device,
        lengths=tuple(args.lengths),
        repeats=args.repeats,
    )

    out = Path(args.out) if args.out else Path("data/latency") / _slug(run.sku)
    run.write(out)
    usable = sum(1 for m in run.measurements if m.ok)
    print(f"\n{usable}/{len(run.measurements)} measured, wrote {out}")
    print(f"use it with: basalt verify <cubin> --latencies {out}")
    return 0


def _slug(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in name]
    return "".join(keep).strip("-").replace("--", "-") + ".json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="basalt", description="a SASS assembler for sm_120")
    ap.add_argument("--version", action="version", version=f"basalt {__version__}")
    ap.add_argument("--cuda-bin", default=None, help="directory holding ptxas and nvdisasm")
    ap.add_argument("--arch", default="sm_120a", help="target architecture (default sm_120a)")

    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="verify the toolchain and both oracles")

    b = sub.add_parser("build-isa", help="harvest and probe, then write the ISA database")
    b.add_argument("-o", "--out", default=DEFAULT_DB)
    b.add_argument("--harvest-out", default=None, help="also write the raw harvest record")
    b.add_argument("--no-tensor", action="store_true", help="skip the tensor-core corpus")

    q = sub.add_parser("isa", help="query the ISA database")
    q.add_argument("mnemonic", nargs="?", help="show one form in detail")
    q.add_argument("--db", default=DEFAULT_DB)
    q.add_argument("--opcode", help="list every form of one opcode")
    q.add_argument("--stats", action="store_true", help="print coverage")

    m = sub.add_parser("measure", help="measure instruction latency on a real device")
    m.add_argument("-o", "--out", default=None, help="output path (default names the SKU)")
    m.add_argument("--device", type=int, default=0, help="CUDA device ordinal")
    m.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="chain lengths to fit the slope across",
    )
    m.add_argument("--repeats", type=int, default=7, help="launches per length; the minimum wins")

    ps = sub.add_parser(
        "probe-stalls",
        help="find the required stall per opcode by breaking programs on purpose",
    )
    ps.add_argument("--opcode", default=None, help="probe only this opcode")
    ps.add_argument("--device", type=int, default=0)
    ps.add_argument("--links", type=int, default=8, help="chain length to build")
    ps.add_argument("--repeats", type=int, default=12, help="launches per candidate stall")

    v = sub.add_parser("verify", help="check a cubin's control bits for data hazards")
    v.add_argument("cubin", help="path to a cubin, whatever produced it")
    v.add_argument("--latencies", default=None, help="measured latency JSON to overlay")
    v.add_argument(
        "--observed",
        default=None,
        help="mined per-pair stall data (defaults to the committed file if present)",
    )
    v.add_argument("--limit", type=int, default=25, help="maximum hazards to print")
    v.add_argument("--all", action="store_true", help="include informational findings")
    v.add_argument("--strict", action="store_true", help="exit non-zero when a hazard is found")

    args = ap.parse_args(argv)
    return {
        "doctor": _doctor,
        "build-isa": _build_isa,
        "isa": _isa,
        "measure": _measure,
        "probe-stalls": _probe_stalls,
        "verify": _verify,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
