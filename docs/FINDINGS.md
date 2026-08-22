# Findings

Measurements of sm_120 made with basalt, each with the command that reproduces it and the
evidence it rests on. Where a result is uncertain or not claimed, it says so.

All figures below are from an **NVIDIA GeForce RTX 5070 Ti** (sm_120, 70 SMs) with
`ptxas` **V13.3.73**. Published characterisation of this architecture has used other parts,
so a figure differing on a different SKU is interesting rather than contradictory.

---

## 1. A stall count of zero is a safe encoding, not zero cycles

The `stall` field is four bits. Value 0 does **not** mean "issue the next instruction
immediately". It is a distinct encoding that waits for outstanding results as well as
elapsed cycles.

Measured over a 128-link dependent `IMAD` chain, patching the control bits of every link
directly and comparing the answer against a reference:

| `stall` | cycles per instruction | result |
| ---: | ---: | :--- |
| **0** | **36.85** | **correct** |
| 1 | 4.88 | wrong |
| 2 | 4.88 | wrong |
| 3 | 5.88 | wrong |
| 4 | 6.88 | correct |
| 8 | 10.88 | correct |
| 15 | 18.02 | correct |

So a zero stall costs roughly nine times a correctly scheduled instruction and is always
safe, while 1 to 3 corrupt silently.

**Why it matters beyond the encoding.** This is why `ptxas -O0` emits an entirely zeroed
control word, with no stalls and no scoreboards anywhere, and the code still computes the
right answer about nine times slower. It also means unscheduled output cannot be used to
learn anything about what a dependency requires, and that a checker reading 0 as zero
cycles will report correct programs as broken.

```bash
python -m basalt.cli verify path/to/O0.cubin --latencies data/latency/rtx-5070-ti.json
```

## 2. The checker agrees with the vendor compiler across the whole corpus

Every kernel `ptxas` builds from the generated corpus is compiled and verified.
Its scheduling is the reference, so an error on any of them is basalt's fault.

| | |
| :--- | ---: |
| Kernels compiled | 317 |
| Dependencies checked | 5,423 |
| Errors | **0** |
| Kernels with warnings | 5 |

The five warnings are opcodes whose latency is still assumed rather than
measured, plus one late-read case, and they are reported as warnings for exactly
that reason.

This is not a formality. Every modelling error basalt has made was found here or
by the smaller version of it, never by reasoning about the architecture:
scoreboards treated as flags rather than counters, a wait required from every
consumer rather than from any instruction, a guard predicate read as an opcode,
a scoreboard ignored because the producer was fixed-latency, `VOTEU` classified
as completing out of order, and a stall requirement treated as a property of the
producer when it belongs to the pair.

```bash
pytest -m slow      # runs the control; it also runs in CI on every push
```

## 3. Understalling corrupts silently, and basalt predicts exactly when

The premise of the project, checked directly rather than argued. For each encodable stall on
a dependent producer, basalt's static verdict is compared against what the silicon computes:

| `stall` | basalt says | hardware computes | agree |
| ---: | :--- | :--- | :--- |
| 0 | clean | correct | yes |
| 1 | hazard | **wrong** | yes |
| 2 | hazard | **wrong** | yes |
| 3 | hazard | **wrong** | yes |
| 4 | clean | correct | yes |
| 5 | clean | correct | yes |
| 6 | clean | correct | yes |
| 7 | clean | correct | yes |

No crash, no fault, no warning at any of the wrong rows. The wrong answer is also
deterministic rather than a race: every repeat produces the same incorrect value.

This is held as a test (`tests/test_gpu.py::TestVerdictsMatchHardware`), so a change that
breaks the agreement fails the suite.

## 4. Required stall, by three independent methods

Three ways of asking the question, which do not always agree because they are not quite the
same question.

- **Chain timing** measures `max(latency, initiation interval)`, so a rate-limited unit reads
  high.
- **What `ptxas` schedules** is an upper bound: the compiler may be cautious.
- **Fault injection** measures the requirement itself, by shortening the gap until the answer
  changes.

| Instruction | Chain timing | `ptxas` leaves | Fault injection | Reading |
| :--- | ---: | ---: | ---: | :--- |
| `IMAD` | 4 | 4 | **4** | all three agree |
| `IADD3` | 4 | 4 | **4** | all three agree |
| `FFMA` | 4 | 4 | **4** | all three agree |
| `FADD` | 4 | 4 | **4** | all three agree |
| `FMUL` | 4 | 4 | **4** | all three agree |
| `LOP3` | 4 | 4 | **4** | all three agree |
| `SHF` | 4 | 4 | **4** | all three agree |
| `I2FP` | 24 (with `F2I`) | 6 | not established | see below |
| `MUFU` | 44 | n/a | scoreboarded | result takes 44, covered by a scoreboard |
| `POPC` | 18 | n/a | scoreboarded | same |
| `DADD` | 64 | 64 | scoreboarded | see below |
| `DFMA` | 64 | 64 | scoreboarded | see below |

```bash
python -m basalt.cli measure -o data/latency/your-card.json   # chain timing
python -m basalt.cli probe-stalls                             # fault injection
```

**On the fp64 rows.** The 64-cycle figure is corroborated twice over: `ptxas` covers a `DFMA`
dependency by padding with NOPs at stall 15 and accumulating exactly 64 cycles, matching the
independently timed figure to the cycle. It also signals a scoreboard on the same
instruction, which is belt and braces, and it is the scoreboard that makes shortening the
stalls harmless. That is why fault injection reports the pair as scoreboarded rather than
returning a number.

**On `MUFU` and `POPC`.** A deterministic latency and a stall-covered latency are different
things. `MUFU` produces a perfectly linear 44 cycles under timing, but `ptxas` signals a
scoreboard on it and the dependent instruction waits on that scoreboard, so the stall does
not have to carry the dependency. basalt keeps it classified as variable for that reason.

## 5. Tensor cores: throughput and requirement are far apart

Timed with a dependent chain that accumulates through the D operand, so each
`mma.sync` cannot issue until the previous one has written the accumulator back.
`ptxas` emits these back to back with nothing between them, so the slope is the
instruction and nothing else.

| Instruction | Cycles per instruction, one warp | R² |
| :--- | ---: | ---: |
| `HMMA.16816.F32` (f16) | 34.52 | 0.99999 |
| `HMMA.16816.F32.BF16` | 34.44 | 1.00000 |
| `HMMA.1688.F32.TF32` | 34.37 | 1.00000 |
| `IMMA.16832.S8.S8` | 26.86 | 1.00000 |
| `QMMA.16832.F32.E4M3.E4M3` | 34.36 | 1.00000 |
| `QMMA.16832.F32.E5M2.E5M2` | 34.41 | 1.00000 |
| `QMMA.16832.F32.E3M2.E3M2` | 34.48 | 1.00000 |
| `QMMA.16832.F32.E2M1.E2M1` | 34.50 | 1.00000 |

Two things stand out.

**The low-precision format does not change this number.** E4M3, E5M2, E3M2 and
E2M1 all land within a tenth of a cycle of each other and of f16. Whatever FP4
buys on this part, it is not a shorter dependent-accumulate interval.

**Integer is faster than float here**, at 26.9 against 34.4.

**These are not latencies.** `ptxas` schedules the same back-to-back chain with
`stall=11`, and shortening it further still computes the right answer, so the
figure above is the interval at which one warp can push dependent matrix
operations through the unit rather than the time a result takes to appear. The
distinction matters: a checker that treated 34 as the required stall would
reject correct code. It is recorded here as throughput, and the number is not
written into the latency model.

## 6. The instruction database is usable, not just readable

Knowing which bits moved an operand describes an encoding. Being able to write a
value into those bits and get it back is what an assembler needs, and it does not
follow automatically: a table built by observation can be a description that
happens to be wrong.

So every measured field is written through and read back, across five values
rather than one, since a single value can agree by accident.

| | |
| :--- | ---: |
| Forms checked | 269 |
| Register-bearing operand slots | 726 |
| Slots that behave as measured | **710 (97.8%)** |
| Forms with every register slot controllable | 246 |

The 207 remaining slots hold something other than a register, a branch target,
an immediate, a named special register such as `SR_CLOCKLO`, and are counted
apart rather than reported as failures, because reading a register number back
out of them is the wrong question.

Sixteen forms still have a slot that does not behave, mostly a carry or
predicate input in the last position on extended-precision forms. Those are
listed rather than rounded away.

```bash
python -m basalt.cli validate-isa --show
```

## 7. The stall field cannot express a long latency

Four bits, so 15 is the largest gap a single instruction can request. Any requirement above
that must be covered by accumulating stalls across several instructions, or by a scoreboard.
`ptxas` does both: for a 64-cycle fp64 dependency it emits

```
DFMA R4, R4, R6, R4     stall=15  wr=1
NOP                     stall=15
NOP                     stall=15
NOP                     stall=15
NOP                     stall= 4          <- 15+15+15+15+4 = 64
DFMA R4, R6, R4, R4     stall=15  wait=0x02
```

Four NOPs whose only purpose is to spend cycles.

## 8. What is deliberately not claimed

Stated so the boundary of the evidence is visible.

- **`I2FP` and `F2I` cannot be separated by timing.** A conversion cannot feed the next link
  of a dependent chain without converting back, so the chain always contains one of each. The
  24-cycle figure is the pair, and it is recorded in a separate `composite` section rather
  than halved and presented as a per-instruction latency. Fault injection cannot separate them
  either, because the round trip is idempotent and the chain reaches a fixed point.
- **Only one SKU has been measured.** Everything here is an RTX 5070 Ti. Whether these figures
  hold across sm_120 parts with different SM counts is exactly the sort of thing that should
  not be assumed, and basalt records the part alongside every measurement so a second card can
  be compared rather than merged.
- **Most opcodes still carry assumed latencies.** The model marks them as such, and a hazard
  derived from an assumed number is reported as a warning rather than an error. The difference
  between a lead and a finding is where the number came from.
- **A test that cannot detect corruption proves nothing.** An early version of the injection
  probe multiplied by 1.0000001, so a stale read rounded back to the same float and `FMUL`
  appeared to need only one cycle. Every probe now runs a sensitivity control first: a chain
  one link shorter must produce a different answer, established independently of the stall
  sweep. Without that control, "no value was unsafe" is ambiguous between "every value is
  genuinely safe" and "this kernel cannot tell", which are opposite claims.
- **`POPC` and `I2FP` fail that control**, because their chains reach a fixed point: `popc`
  of a `popc` stops changing, and an integer round trip through float is idempotent after the
  first conversion. An earlier run reported `I2FP` as requiring 4 cycles; the control
  retracted it, and it is listed as not established rather than quietly kept.

## 9. Corrections made along the way

Kept because a method is only as trustworthy as its error log.

- Scoreboards are counters, not flags. An early rule treated a second signal on the same
  scoreboard as a hazard; several producers sharing one is ordinary.
- A wait by any intervening instruction satisfies a dependency for everything downstream.
  Requiring each consumer to carry its own wait produced eight false findings in a
  thirty-two instruction kernel.
- A guard predicate is printed before the mnemonic, so reading the first token as the opcode
  silently misparses every predicated instruction.
- A scoreboard covers a dependency whatever the producer's latency class. Checking stalls
  only for fixed-latency producers missed that `ptxas` scoreboards fp64.

Each of these was caught by the positive control: the vendor compiler's own output must
verify clean, and every one of them made it fail.
