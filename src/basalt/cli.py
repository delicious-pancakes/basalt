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
        # an instruction the listing gave no encoding for has nothing to decode
        back = decode_words(tc, [i.word for i in insns if i.word is not None], arch=arch_raw)
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
# The measured latencies for the one card this has been run on. Used when no
# `--latencies` is given, so a command run from a checkout gets the measurements
# rather than the assumptions without having to be told.
DEFAULT_LATENCIES = "data/latency/rtx-5070-ti.json"
DEFAULT_ISA = "data/isa/sm_120a.json"


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


def _validate_isa(args: argparse.Namespace) -> int:
    """Check that the measured operand fields can actually be written through."""
    from pathlib import Path

    from .isa.database import IsaDatabase
    from .isa.validate import validate_database
    from .toolchain import ToolchainError, find_toolchain

    try:
        tc = find_toolchain(args.cuda_bin)
    except ToolchainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    path = Path(args.db)
    if not path.exists():
        print(f"error: {path} not found; run `basalt build-isa` first", file=sys.stderr)
        return 1

    db = IsaDatabase.read(path)
    summary = validate_database(tc, db, limit=args.limit)
    print()
    print(summary.summary())
    print(f"  forms with every register slot controllable: {len(summary.usable)}")
    print(f"  forms with a slot that does not behave:      {len(summary.broken)}")

    if summary.broken and args.show:
        print()
        for r in summary.broken:
            print(f"  {r.mnemonic:<30} ok={r.controllable} bad={r.uncontrollable}")

    return 0 if not summary.broken or not args.strict else 2


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


def _mine_stalls(args: argparse.Namespace) -> int:
    """Mine the compiler's own scheduling for per-pair stall requirements."""
    from pathlib import Path

    from .toolchain import ToolchainError, find_toolchain
    from .verify.observed import mine_corpus

    try:
        tc = find_toolchain(args.cuda_bin)
    except ToolchainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    observed = mine_corpus(tc, arch=args.arch)
    out = Path(args.out)
    observed.write(out)
    print()
    print(observed.summary())
    print(f"wrote {out}")

    if args.show:
        print()
        for name, ev in sorted(observed.by_producer.items()):
            if ev.trusted:
                print(f"  {name:<10} min {ev.minimum:2d}   from {ev.observations} observations")
    return 0


def _assemble(args: argparse.Namespace) -> int:
    """Encode one instruction, and optionally read it back.

    Reading it back is not decoration. The assembler either reproduces the bytes
    the vendor compiler emits or refuses, and `--verify` is how that claim is
    checked from the outside: the word goes through `nvdisasm` and the text that
    comes out is printed beside the text that went in.
    """
    import sys as _sys
    from pathlib import Path

    from .asm.assemble import Assembler, AssemblyError
    from .isa.database import IsaDatabase

    database = Path(args.isa)
    if not database.is_file():
        print(f"error: {database} not found; run `basalt build-isa`", file=_sys.stderr)
        return 1

    if args.cubin:
        return _assemble_cubin(args, database)

    text = args.text if args.text else _sys.stdin.read()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        print("error: nothing to assemble", file=_sys.stderr)
        return 1

    assembler = Assembler(IsaDatabase.read(database))
    toolchain = None
    if args.verify:
        from .toolchain import ToolchainError, find_toolchain

        try:
            toolchain = find_toolchain(args.cuda_bin)
        except ToolchainError as exc:
            print(f"error: {exc}", file=_sys.stderr)
            return 1

    failures = 0
    for line in lines:
        try:
            word = assembler.assemble(line)
        except AssemblyError as exc:
            print(f"  refused  {line}")
            print(f"           {exc}")
            failures += 1
            continue
        print(f"  {word.value:032x}  {line}")
        if toolchain is not None:
            from .disasm import decode_word

            back = decode_word(toolchain, word)
            reads = f"{back.mnemonic} {back.operands}".strip() if back else "(not decodable)"
            agrees = "same" if reads == line else "DIFFERENT"
            print(f"           reads back as {reads}  [{agrees}]")
            if reads != line:
                failures += 1

    return 1 if failures else 0


def _assemble_cubin(args: argparse.Namespace, database) -> int:
    """Assemble a whole cubin and say how much of it came back identical.

    This is the assembler's own control. Every instruction goes out as text and
    comes back as bits, and the bits have to be the ones the vendor compiler
    emitted. Anything that cannot be encoded is listed with its reason rather
    than approximated, so the number reported is coverage and not a score.
    """
    import sys as _sys
    from pathlib import Path

    from .asm.assemble import assemble_program
    from .disasm import disassemble_program
    from .isa.database import IsaDatabase
    from .toolchain import ToolchainError, find_toolchain

    try:
        tc = find_toolchain(args.cuda_bin)
    except ToolchainError as exc:
        print(f"error: {exc}", file=_sys.stderr)
        return 1

    target = Path(args.cubin)
    if not target.is_file():
        print(f"error: {target} not found", file=_sys.stderr)
        return 1

    program = disassemble_program(tc, target)
    if not program.instructions:
        print(f"error: nothing disassembled from {target}", file=_sys.stderr)
        return 1

    result = assemble_program(program, IsaDatabase.read(database))
    exact = wrong = 0
    for instruction, got in zip(program.instructions, result.words, strict=True):
        if instruction.word is None or got is None:
            continue
        if got.value == instruction.word.value:
            exact += 1
        else:
            wrong += 1

    total = exact + wrong + len(result.refused)
    print(f"{target}")
    print(f"  {exact}/{total} instructions reassembled bit-identically")
    if wrong:
        print(f"  {wrong} assembled to different bytes, which is a defect")
    for index, text, reason in result.refused[: args.limit if hasattr(args, "limit") else 10]:
        print(f"  refused #{index}: {text}")
        print(f"           {reason}")
    return 1 if wrong else 0


def _schedule(args: argparse.Namespace) -> int:
    """Throw away a cubin's control bits, compute new ones, and check them.

    The verifier answers whether a schedule is safe. This answers what a safe
    schedule would be, from the same measurements, and then hands the answer
    straight back to the verifier. That round trip is the point: a scheduler and
    a checker that disagree have found something, and one that agrees with
    itself has only proved it is consistent.

    Consistency is not correctness here, and the tool says so. Both halves read
    the same latency model, so a wrong entry satisfies both at once. Only the
    silicon is independent, which is what `scripts/roundtrip_corpus.py` is for.
    """
    from pathlib import Path

    from .asm.cubin import Cubin
    from .disasm import disassemble_program
    from .sched.scheduler import issue_cycles, schedule_program
    from .toolchain import ToolchainError, find_toolchain
    from .verify.hazards import verify_program
    from .verify.latency import DEFAULT_MODEL, LatencyModel
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
    elif (measured := Path(DEFAULT_LATENCIES)).is_file():
        model = LatencyModel.assumed().overlay(measured)

    observed = None
    if args.observed:
        observed = ObservedStalls.read(Path(args.observed))
    elif (default := Path(DEFAULT_OBSERVED)).is_file():
        observed = ObservedStalls.read(default)

    program = disassemble_program(tc, target)
    if not program.instructions:
        print(f"error: nothing disassembled from {target}", file=sys.stderr)
        return 1

    result = schedule_program(program, model, observed=observed)
    print(f"{target}")
    print(f"  {result.summary()}")

    # what the correctness costs, stated rather than left for someone to find
    before = issue_cycles([i.word for i in program.instructions], program.instructions)
    after = issue_cycles(result.words, program.instructions)
    if before:
        print(f"  issue cycles: {before} as compiled, {after} as scheduled ({after / before:.2f}x)")
    for note in result.out_of_scoreboards:
        print(f"  unallocatable: {note}")
    for note in result.unplaceable[:5]:
        print(f"  did not fit: {note}")

    destination = Path(args.output) if args.output else target.with_suffix(".resched.cubin")
    cubin = Cubin.load(target)
    for slot, word in enumerate(result.words):
        if program.instructions[slot].word is not None:
            cubin.write_word(slot, word)
    cubin.save(destination)
    print(f"  wrote {destination}")

    written = disassemble_program(tc, destination)
    if len(written.instructions) != len(program.instructions):
        print(
            f"  error: the result disassembled to {len(written.instructions)} instructions "
            f"where the input had {len(program.instructions)}, so it cannot be checked back",
            file=sys.stderr,
        )
        return 1
    report = verify_program(written, model, observed=observed)
    print(f"  checked back: {report.summary()}")
    for hazard in report.hazards[:10]:
        print(f"    {hazard.describe()}")

    if args.strict and not (report.ok and result.ok):
        return 1
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
    elif (measured := Path(DEFAULT_LATENCIES)).is_file():
        # the committed measurements beat the assumptions, and a checkout has
        # them, so a bare `verify` should not quietly use the weaker model
        model = LatencyModel.assumed().overlay(measured)

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
        trusted = sum(1 for e in observed.by_producer.values() if e.trusted)
        print(
            f"  pair data: {len(observed.by_pair)} pairings from {observed.kernels} kernels, "
            f"{trusted} producers with enough observations to use"
        )
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
        board=args.board,
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
    ap = argparse.ArgumentParser(
        prog="basalt",
        description="the correctness layer for sm_120 machine code",
    )
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
    m.add_argument(
        "--board",
        default="",
        help='board model, e.g. "Gigabyte RTX 5070 Ti EAGLE OC". the driver reports the GPU '
        "and not the partner board, and a run is easier to reproduce when the card is named "
        "exactly. it does not change the numbers: latencies are in cycles",
    )

    ps = sub.add_parser(
        "probe-stalls",
        help="find the required stall per opcode by breaking programs on purpose",
    )
    ps.add_argument("--opcode", default=None, help="probe only this opcode")
    ps.add_argument("--device", type=int, default=0)
    ps.add_argument("--links", type=int, default=8, help="chain length to build")
    ps.add_argument("--repeats", type=int, default=12, help="launches per candidate stall")

    va = sub.add_parser(
        "validate-isa",
        help="check the measured operand fields can be written through, not just read",
    )
    va.add_argument("--db", default=DEFAULT_DB)
    va.add_argument("--limit", type=int, default=None, help="validate only the first N forms")
    va.add_argument("--show", action="store_true", help="list the forms that failed")
    va.add_argument("--strict", action="store_true", help="exit non-zero if any form fails")

    ms = sub.add_parser(
        "mine-stalls",
        help="learn per-pair stall requirements from what the compiler schedules",
    )
    ms.add_argument("-o", "--out", default=DEFAULT_OBSERVED)
    ms.add_argument("--show", action="store_true", help="print the per-producer minimums")

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

    s = sub.add_parser(
        "schedule",
        help="assign a cubin's control bits from scratch and check the result",
    )
    s.add_argument("cubin", help="path to a cubin, whatever produced it")
    s.add_argument("-o", "--output", default=None, help="write the rescheduled cubin here")
    s.add_argument("--latencies", default=None, help="measured latency JSON to overlay")
    s.add_argument(
        "--observed",
        default=None,
        help="mined per-pair stall data (defaults to the committed file if present)",
    )
    s.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero unless the result verifies clean",
    )

    a = sub.add_parser(
        "assemble",
        help="encode SASS text, and check it against what the vendor emits",
    )
    a.add_argument(
        "text",
        nargs="?",
        help='one instruction, e.g. "IMAD R7, R2, R6, RZ". omit to read stdin',
    )
    a.add_argument("--isa", default=DEFAULT_ISA, help="instruction database to encode against")
    a.add_argument(
        "--cubin",
        default=None,
        help="assemble a whole cubin instead: disassemble it, encode every instruction with "
        "its labels resolved, and report how much came back bit-identical",
    )
    a.add_argument(
        "--verify",
        action="store_true",
        help="disassemble the result and show what it reads back as",
    )

    args = ap.parse_args(argv)
    return {
        "doctor": _doctor,
        "build-isa": _build_isa,
        "isa": _isa,
        "measure": _measure,
        "validate-isa": _validate_isa,
        "mine-stalls": _mine_stalls,
        "probe-stalls": _probe_stalls,
        "verify": _verify,
        "schedule": _schedule,
        "assemble": _assemble,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
