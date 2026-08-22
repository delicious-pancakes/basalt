# Roadmap

What basalt is building toward, what is done, and what is deliberately out of scope. Updated as
stages land. Anything claimed here as done has a command that demonstrates it.

## The position

basalt is a **checker**, not a code generator.

Tools that emit sm_120 machine code *assign* the scheduling control word from a latency model.
Nothing publicly available verifies the result afterwards, and the latency models in use were
validated against a single GPU part.

That matters because sm_120 has no hardware interlock on fixed-latency instructions. The
hardware does not validate the scheduling control word; it trusts whatever produced it. A stall
count shorter than the latency of the value it consumes does not fault and does not stall. It
reads a stale register and returns a wrong answer at full speed, silently, every time.

basalt is the check. It works on machine code regardless of what produced it, including the
vendor compiler's own output, which is what makes it useful rather than self-referential.

## Stages

| # | Stage | Delivers | State |
| :-- | :--- | :--- | :--- |
| 1 | Oracles | `ptxas`/`nvdisasm` round-trip harness, raw-word probe | **done** |
| 2 | Harvest | PTX corpus, encoding extraction at scale | **done** |
| 3 | Field inference | Per-bit roles by differential mutation | **done** |
| 4 | ISA database | Grounded, provenanced instruction forms | **done** |
| 5 | Hazard model | Def-use analysis over decoded programs | **done** |
| 6 | Verifier | Static hazard checking over any cubin | **done** |
| 7 | Latency measurement | Per-SKU instruction latency on real silicon | **done**, one SKU |
| 8 | Cross-block analysis | A real control-flow graph, so definitions survive branches | **done** |
| 8b | Fault injection | Required stall measured by breaking programs on hardware | **done** |
| 9 | Field validation | Prove the measured fields can be written through | **done** |
| 9b | Cross-check | basalt's ISA model against independently derived tables | planned |
| 10 | Audit | Every public sm_120 SASS kernel, ptxas as the control | planned |
| 11 | Scheduler | Assign the control bits, not only check them | **experimental** |
| 12 | Assembler | SASS text to the instruction word | **working** |

The scheduler discards every control bit `ptxas` produced and computes its own, then hands the
result back to the verifier and then to the GPU, for every kernel the corpus generates.
every one of the 314 comparable ones comes out byte-identical to the vendor schedule. The rest are
named in the findings rather than summarised, and that count is what the work is measured
against: it was 246 when the control was first run, and every model correction since came
out of watching it move.

The assembler encodes SASS text back into the instruction word, and its standard is the
vendor's own bytes: assembling every corpus kernel as a whole program, with its labels
resolved, reproduces 8,351 of 8,560 instructions bit-identically and none to anything else.
That second number is a test pinned at zero.

Whole programs rather than lone instructions because a branch cannot be assembled alone.
Its field holds the distance to the destination, so the same text encodes differently in
every kernel it appears in. That field was solved from real kernels, the label table giving
the destination and the word giving the bits, and it is split across bits 16..23 and 34..81
with the value scaled by four, which is why searching contiguous runs finds nothing.

What it still declines is a handful of fields the prober could only partly attribute, and a
few operand shapes the harvest has not sampled. Both are coverage rather than correctness,
and both surface as a refusal with a reason rather than a wrong word.

Stages 1 to 6 need no GPU and run in CI. Stage 7 needs an sm_120 card, and only the
measurement step does; the numbers it produces are a checked-in file everyone else reads.

## Findings this is positioned to produce

Each of these yields a result whether the answer is positive or negative, which is the point.
A tool can be ignored; a measurement cannot.

1. **ISA disagreements.** basalt derives its instruction model by differential bit probing.
   Other tools extract tables. Two independent derivations of one ISA that disagree mean one is
   wrong, and finding out which is a contribution either way.
2. **Per-SKU latency.** Published sm_120 characterisation covers single parts. If latency varies
   across sm_120 SKUs then scheduling tuned on one part is unsound on the others, which would be
   a significant result. If it does not vary, that is a useful negative result that lets every
   existing scheduler claim portability with evidence.
3. **Unsafe control bits in shipped kernels.** Hand-written SASS is now in production use. The
   verifier either finds hazards there or establishes that it does not.

## The control that keeps the audit honest

> [!IMPORTANT]
> The verifier runs against `ptxas` output first, always. The vendor compiler's scheduling is
> the reference: if basalt flags it, basalt is wrong, and that is chased to root cause before
> any finding about anyone else's code is published. An audit tool with no control is an
> opinion generator.

## Out of scope

Stated so the boundary is deliberate rather than accidental.

- **Competing on code generation.** basalt must work on *other* tools' output, not only its own.
  The stage 10 assembler exists to generate test programs, not to win a comparison.
- **Optimising schedules.** Making SASS faster is a separate problem with existing research
  behind it. basalt answers whether a schedule is *safe*, not whether it is *good*.
- **Architectures other than sm_120.** The method generalises; the measurements do not. Claiming
  coverage that has not been measured on the silicon in question is the exact failure basalt
  exists to catch.
- **Anything requiring NVIDIA source, headers, or decompilation.** See
  [`NOTICE`](../NOTICE) and [`CONTRIBUTING.md`](../CONTRIBUTING.md).
