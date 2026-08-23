<img src="assets/header-roadmap.svg" alt="basalt Roadmap" width="100%" />


What basalt is building toward, what is done, and what is deliberately out of scope. Updated as
stages land. Anything claimed here as done has a command that demonstrates it.

## The position

basalt is a **checker** first and a code generator second. The code generators are not the
first for this architecture; the checker is, and it is the only one of the three measured
against machine code it did not produce.

Tools that emit sm_120 machine code *assign* the scheduling control word from a latency model,
and the ones that check themselves do it by running their own kernels and seeing that the
answers come out right. Nothing publicly available reads machine code it did not produce and
says whether that code's control bits are safe, which is the question anyone holding a cubin
actually has.

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
| 9b | Cross-check | The ISA model derived twice, from two compiler releases | **done**, 0 operand fields differ |
| 10 | Audit | Every sm_120 kernel NVIDIA ships, held out of every table | **done**, 0 errors, 13 model bugs |
| 11 | Scheduler | Assign the control bits, not only check them | **done**, every field derived |
| 12 | Assembler | SASS text to the instruction word | **done**, 0 wrong, 67 refused by name |

The audit is the stage that changed the others. Everything before it ran on kernels basalt
compiled itself, which cannot fail: the requirement table is mined from the compiler's own
scheduling, so the tightest gap it left *is* the floor, by construction, for exactly that
code. Pointed at 250 kernels out of NVIDIA's JPEG decoder it reported 6,593 errors and every
one was basalt's. Eight corrections took that to zero, and zero over one library was not
evidence either: widening to three libraries and 5.2 million instructions took it back to 940
and produced five more. Thirteen in total, none of them NVIDIA's, and the held-out set now
verifies with 0 errors over 2,762 kernels and 10,218,030 dependencies, all 2,762 fully
analysed. Finding 32.

The scheduler discards every control bit `ptxas` produced and computes its own, then hands the
result back to the verifier and then to the GPU, for every kernel the corpus generates. All
439 comparable ones come out byte-identical to the vendor schedule, at all three optimisation
levels. The two excluded read the clock and the grid id, so they do not agree with themselves
either. That count is what the work is measured against: it was 246 when the control was first
run, and every model correction since came out of watching it move.

Every field is derived, and so is every number behind them. Read barriers guard an
instruction that has not finished taking its operands at issue, and for most of this project
basalt copied them out of the schedule it was replacing, which meant it could not have
scheduled a program nobody had compiled. Characterising all 299 of them in the corpus gave
the rule it places them by now (finding 25), and the test that it is a real rule rather than
a plausible one is that taking the derived barriers away makes a kernel return the wrong
answer on the card.

The last inherited quantity went with them. Inside the window a barrier covers, basalt used
to hold the vendor's own stalls as a floor; it now holds the issue rate mined from every
consecutive pairing in the corpus, which is the same 4 cycles for a run of loads and is a
number that exists for a program the vendor never saw.

The assembler encodes SASS text back into the instruction word, and its standard is the
vendor's own bytes: assembling every corpus kernel as a whole program, with its labels
resolved, reproduces 59,693 of 59,760 instructions bit-identically across four optimisation
levels, and none to anything else. That second number is a test pinned at zero.

| `ptxas` level | Exact | Refused | Wrong |
| :--- | ---: | ---: | ---: |
| -O0 | 22,714 of 22,752 | 38 | **0** |
| -O1 | 12,189 of 12,208 | 19 | **0** |
| -O2 | 12,395 of 12,400 | 5 | **0** |
| -O3 | 12,395 of 12,400 | 5 | **0** |

Whole programs rather than lone instructions because a branch cannot be assembled alone.
Its field holds the distance to the destination, so the same text encodes differently in
every kernel it appears in. That field was solved from real kernels, the label table giving
the destination and the word giving the bits, and it is split across bits 16..23 and 34..81
with the value scaled by four, which is why searching contiguous runs finds nothing.

What it still declines is five instructions at `-O3`, and they are the same three shapes at
every level: a `RET` or `WARPSYNC` whose register shares a field with its branch target and
was never isolated from it, the invert bit on a `BRA` predicate, and a `VIMNMX` immediate
form the harvest has not sampled. All coverage rather than correctness, and all surfacing as
a refusal that names the field rather than as a wrong word.

Stages 1 to 6 need no GPU and run in CI. Stage 7 needs an sm_120 card, and only the
measurement step does; the numbers it produces are a checked-in file everyone else reads.

## Findings this is positioned to produce

Each of these yields a result whether the answer is positive or negative, which is the point.
A tool can be ignored; a measurement cannot.

1. **ISA disagreements.** basalt derives its instruction model by differential bit probing.
   Other tools extract tables. Two independent derivations of one ISA that disagree mean one is
   wrong, and finding out which is a contribution either way.
2. **Per-SKU latency.** *Answered, and it is the negative result.* `ptxas` compiles the whole
   corpus to byte-identical code, control words included, for all six targets in this family
   including `sm_121`, which is a different chip. The required schedule is a property of the
   architecture rather than of the part, so a scheduler tuned on one part is sound on the
   others, with evidence rather than hope. Finding 28.
3. **Unsafe control bits in shipped kernels.** *Answered, and the answer is about basalt.*
   2,473 sm_120 cubins out of CUDA 13.3.1, 844 MB of device code nobody here compiled. The
   first run reported 6,593 errors in 250 nvjpeg kernels and every one was a hole in basalt's
   model, eight of them, each invisible on the corpus by construction. Widening the held-out
   set to three libraries and 5.2 million instructions found five more. After all thirteen:
   zero errors over 2,762 kernels and 10,218,030 dependencies, on libraries held out of every
   table the checker reads. Finding 32.

## The control that keeps the audit honest

> [!IMPORTANT]
> A table mined from the code it is then checked against cannot fail. That is what stage 10
> found, so the requirement is mined from one set of libraries and the audit reports on
> another, and the corpus positive control still runs first.

> [!IMPORTANT]
> The verifier runs against `ptxas` output first, always. The vendor compiler's scheduling is
> the reference: if basalt flags it, basalt is wrong, and that is chased to root cause before
> any finding about anyone else's code is published. An audit tool with no control is an
> opinion generator.

## Out of scope

Stated so the boundary is deliberate rather than accidental.

- **Competing on code generation.** basalt must work on *other* tools' output, not only its own.
  The stage 12 assembler exists to generate test programs, not to win a comparison.
- **Optimising schedules.** Making SASS faster is a separate problem with existing research
  behind it. basalt answers whether a schedule is *safe*, not whether it is *good*.
- **Architectures other than sm_120.** The method generalises; the measurements do not. Claiming
  coverage that has not been measured on the silicon in question is the exact failure basalt
  exists to catch.
- **Anything requiring NVIDIA source, headers, or decompilation.** See
  [`NOTICE`](../NOTICE) and [`CONTRIBUTING.md`](../CONTRIBUTING.md).
- **Reordering instructions.** basalt rewrites the control word and leaves the order alone,
  which is what makes a change in behaviour attributable to the bits rather than to the
  schedule. A reordering scheduler is a different tool with a different control.
