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

## 7. fp64 is carried by the scoreboard, and still owes a small stall

`ptxas` covers a dependent `DFMA` pair with both mechanisms at once: 64 cycles of
stall, padded out with maximum-stall NOPs, and a scoreboard signalled on the
producer that the consumer waits on. Which one is actually carrying it can be
settled directly.

| What was changed | Result |
| :--- | :--- |
| Nothing | correct |
| Scoreboards and waits kept, stalls across the fp64 stretch reduced to 1 | correct |
| Stalls kept, the `DADD` write barrier and the store's wait removed | **wrong** |
| Everything kept, the `DADD`'s own stall alone cut from 2 to 1 | **wrong** |

So the scoreboard is what carries the dependency. The 64 cycles are the cost of a
dependent chain, which is a real number and a different question from what
correctness requires.

The last row is the part worth keeping. A wait covers the long, variable part of
the result and not the whole of it: the producer still owes a small stall of its
own, and `ptxas` knows exactly how much. Across the corpus it never schedules a
`DADD` or a `DSETP` below 2 cycles, or a `DFMA` or `DMUL` below 1, however the
consumer waits. basalt mines that minimum per opcode alongside everything else
and applies it wherever a scoreboard is signalled.

Every one of the 41 fp64 instructions in the corpus carries a write scoreboard
and none goes without, so fp64 is modelled as completing out of order rather than
on a fixed schedule.

> [!NOTE]
> **This entry previously concluded the opposite**, that the cycles were
> load-bearing and the scoreboard was belt and braces. The experiment behind it
> reduced every stall in the kernel, including the one on the `LDC.64` that sets
> up the store address, which breaks the kernel on its own and has nothing to do
> with fp64. Confining the change to the fp64 stretch reverses the result. The
> wrong conclusion survived because it agreed with the model already in the code:
> fp64 was classified as fixed latency, so nothing disagreed with it until a
> rescheduled fp64 kernel was run on the GPU and returned 15.84 where the vendor
> returned 20.72.

**A related bug this uncovered.** `DFMA R4, R4, R6, R4` carries no width suffix
anywhere, but fp64 operands occupy register pairs: it writes R4 and R5 and reads
R6 and R7. basalt's operand model widened only on an explicit `.64`, so it saw
half of every fp64 dependency. Silent in the checker, silent in the scheduler,
and found only by running a rescheduled kernel and getting the wrong number.

## 8. The stall field cannot express a long latency

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

## 9. A guard predicate costs two and a half times an ordinary read

`@P1 IMAD` and `SEL R3, R0, R7, P1` both read `P1`. They do not need the same lead.

A guard decides whether the instruction issues at all, so it has to be resolved before
issue. An ordinary source is read later, once the instruction is already going. The gap
between the two is large enough to corrupt a result:

| how the predicate is read | cycles needed from a fixed-latency producer |
| --- | --- |
| as the instruction's guard, `@P1 IMAD` | 13 |
| as a data operand, `SEL ..., P1` | 5 |
| as a general register, for comparison | 5 |

Measured twice, two ways.

**On hardware.** A loop kernel where `ISETP.GT.AND P1, PT, R5, 0x1, PT` feeds
`@P1 IMAD R3, R0, R7, R5`, with the stall on the `ISETP` swept and everything else held:

| stall | result | |
| --- | --- | --- |
| 1 to 5 | 17 or 668, varying between launches | wrong |
| 6 to 8 | 17 or 53 | wrong |
| 9 to 11 | 17 | wrong |
| 12 | 13 | wrong |
| 13 | 2005 | correct |
| 14, 15 | 2005 | correct |

The staircase is the useful part. A single wrong answer could be anything; a value that
climbs monotonically toward the right one as the gap widens, and then stops changing, is a
timing requirement being met.

**Across the corpus.** Every dependent pair in the 317 kernel corpus, split by how the
consumer reads the value, scoreboard-covered pairs excluded because there the stall says
nothing:

| bucket | pairings | samples | median requirement |
| --- | --- | --- | --- |
| guard predicate | 15 | 55 | 13 |
| predicate as data | 7 | 51 | 5 |
| general register | 102 | 785 | 5 |

The discriminating case is `LOP3 -> MOV`, which appears in the corpus both ways: 13 cycles
when `MOV` is guarded by the result, 5 when it reads it as data. Same producer, same
consumer, same distance available, different requirement. So this is a property of issue,
not of the predicate file, and not of the opcode pair.

`ptxas` has clearly always known. It leaves 13 cycles in front of a guard and 5 in front of
a data read, consistently, across every kernel that has both.

### Why this one matters more than its size suggests

Getting it wrong is silent in the dangerous direction. Charging a guard the cheaper
requirement produces a schedule that assembles, loads, launches, returns a plausible
number, and is wrong. Nothing faults. Basalt did exactly this until the sweep above, and
its own checker agreed with its own scheduler the whole time, because both consulted the
same figure.

It also cannot be found by reasoning about the corpus alone. The mined minimum for
`ISETP -> IMAD` is 5, which is correct and useless: it is mined from the unguarded pairs.
Basalt now mines guards under their own key, `ISETP -> @IMAD`, and keeps the two apart in
the per-producer fallback as well. Collapsing them would let one 5 cycle `ISETP -> SEL`
answer for every guard in the program.

Closing this made the loop kernel round-trip, which had been recorded here as a
loop-carried scheduling gap. It was not one. The diagnosis was wrong and the hardware
said so.

## 10. Rescheduling the whole corpus, on the GPU

The scheduler was run over seven hand-written kernels for a long time and passed all
seven. That is a smoke test wearing the clothes of a control.

`scripts/roundtrip_corpus.py` runs it over everything the corpus generates. For each of
the 317 kernels: compile with `ptxas`, discard every control bit it produced, compute new
ones, write them back, and run both versions on the card with identical input. The
rescheduled kernel has to produce the same bytes.

| | |
| :--- | ---: |
| Kernels rescheduled and run | 317 |
| Comparable (the vendor runs here, deterministically, and reproducibly) | 303 |
| **Byte-identical to the vendor schedule** | **303** |
| Wrong | 0 |

Every kernel basalt can be compared on now computes exactly what the vendor's schedule
computes, from control bits it worked out itself. The comparable count moves by one or two
from run to run, because a few kernels' reproducibility depends on what last used the card;
the match count moves with it and the failure count stays at zero.

It holds at three optimisation levels, which is three different vendor schedules to
replace rather than one:

| `ptxas` level | Comparable | Matching |
| :--- | ---: | ---: |
| `-O1` | 301 | 301 |
| `-O2` | 300 | 300 |
| `-O3` | 303 | 303 |

`-O0` is not offered. It emits a zeroed control word, so there is no schedule there to
replace and nothing the comparison would prove.

The first run of this scored 246. Everything between then and now was found by it:

- a guard predicate needing 13 cycles where the same predicate read as data needs 5
  (finding 9),
- a waited-on scoreboard still leaving a residual gap the producer has to cover, and that
  gap being a distance to the consumer rather than the producer's own stall (finding 7),
- the conversion pipe, `POPC` and `FLO` completing out of order rather than on a fixed
  schedule, like fp64 before them,
- requirements having to be keyed on the full mnemonic on both paths, since `I2F.RP` needs
  1 cycle where every other `I2F` needs 2, and `IMAD.WIDE.U32.X` into `IADD` is scheduled at
  3 where plain `IMAD` into `IADD` is scheduled at 5; collapsing either takes a different
  instruction's requirement and wears this one's name,
- a predicated write not killing the previous definition, because `@!P0 FMUL R7, R7, c`
  leaves R7 holding whatever produced it whenever the guard is false,
- an instruction that writes a predicate and a register reading as though it wrote only the
  predicate, so nothing scoreboarded the returned value of `SHFL.IDX PT, R9, ...` or of any
  atomic, and nothing waited for it,
- a variable-latency unit returning results in the order it was given work, so a wait on
  something issued later covers everything that unit still owes: `ptxas` scoreboards the
  second of two consecutive shuffles and waits only on that,
- a wait carried by a predicated instruction not being something a later consumer can lean
  on, since the instruction carrying it may not execute,
- `IMAD.WIDE` writing a register pair, and taking one as its addend, with nothing in the
  mnemonic to say so,
- a call needing everything outstanding to have landed first, because control leaves for
  code this analysis has not read and the callee may use any register,
- the yield bit not being independent of the stall count, which is the one below,
- and the scheduler refusing to allocate a seventh outstanding load instead of sharing a
  scoreboard, which a counter permits and which rejected 45 kernels outright.

None of those were reachable by reasoning, and none were visible to the checker, because
the checker reads the same latency model the scheduler does. A wrong entry satisfies both
at once. That is the whole argument for running the silicon.

### The last one, and why it was the last one

The final two failures were a signed integer divide and a warp-aggregated 64-bit atomic,
and both had resisted every reading of their dependencies. Neither was a dependency
problem.

**The yield bit is not independent of the stall count.** Across the whole corpus `ptxas`
emits a stall of zero with the bit clear 4205 times and set never, and a stall of one with
it set 1123 times and clear never. basalt wrote the stall and left the bit as it found it,
which produced pairs the vendor never emits.

`nvdisasm` refuses them outright, with `undefined value 0x10 for table TABLES_opex_0`. The
GPU does not refuse them. It runs the kernel and returns an answer, which is the worse of
the two outcomes, because nothing complains until something tries to read the result back.

That is also how the bug hid. Two of basalt's own checks reschedule a kernel and hand it
back to the verifier, and a program `nvdisasm` will not read comes back empty. An empty
program has no hazards, so both checks reported clean and had been reporting clean for as
long as the bug existed. They now compare the instruction count first and fail if the
result did not survive the round trip, because a check that passes on nothing is worse than
no check.

Clearing the bit at any stall from 2 upwards is a throughput choice rather than a
correctness one, and `ptxas` is seen doing it at every value in that range, so the rule
basalt follows produces only pairs the vendor also emits.

One reading of the last two failures was implemented before the encoding bug was found, and
is worth recording as rejected. `ptxas` puts a wait on `HFMA2 R4, -RZ, RZ, 0, 0` in the divide,
which reads nothing and writes a constant, so no dependency in the operand list accounts
for it. The obvious explanation is write-after-read: it overwrites R4 while an outstanding
conversion is still reading it.

Teaching the scheduler to wait before overwriting any register an outstanding
variable-latency instruction reads took the corpus from 301 matching down to 293, and made
eight previously correct kernels non-deterministic. SASS carries a separate read-barrier
field for instructions that collect their sources late and `ptxas` uses that rather than
write scoreboards, so treating every scoreboarded instruction as a late reader is not a
conservative approximation of anything. Recorded so the next attempt does not spend the
same afternoon on it.

### What is still excluded

12 kernels that are not runnable by construction, 2 whose vendor output is not
deterministic under 32 threads storing to one address, and any whose result is not
reproducible once something else has used the card.

The first group is the shared and local memory forms. They read shared memory through an
address that has been converted to the global space, which exists to make `ptxas` emit an
`LDS` or an `LDL` and was never meant to execute. Excluding them is not a limitation of the
runner. That last group is why the vendor is
run a second time, after basalt has had the GPU: a kernel reading uninitialised shared
memory is stable until it is not, and its first result is not ground truth. Every `LDSM`
and `MOVMATRIX` kernel read as a basalt failure until that check existed. All three groups
are excluded from the 303 rather than counted as passes.

## 11. What is deliberately not claimed

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

## 12. Corrections made along the way

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
- `VOTEU` was classified as completing out of order. `ptxas` emits it with no scoreboard and
  reads the result on the next instruction, which settles it.
- The required stall belongs to the producer/consumer pair, not to the producer. `IMAD` into
  `IMAD` is scheduled at four cycles and `IMAD` into `IADD` at three.
- fp64 operands are register pairs with nothing in the mnemonic to say so, so half of every
  fp64 dependency was invisible.
- A guard predicate was charged the same as a predicate read as data. It needs thirteen
  cycles against five, and the difference silently corrupts (finding 9).
- A predicated write was treated as killing the previous definition of its register. It
  does not: on the path where the guard is false the earlier producer is what the next
  reader sees, and the dependency on it was being dropped.
- Requirements were keyed on the bare opcode. The modifier decides the number, and
  collapsing forms takes the minimum across them, so `I2F` read as 1 cycle on the strength
  of `I2F.RP` while every other form needs 2.
- The scheduler refused to schedule a seventh outstanding load rather than sharing a
  scoreboard. A scoreboard is a counter, so sharing is permitted and over-synchronises
  slightly; refusing rejected 45 of 317 corpus kernels.
- A scoreboard wait was treated as covering a dependency completely. It does not: the
  producer still owes a per-opcode minimum stall, 2 cycles for `DADD`, and one cycle less
  is silently wrong.
- fp64 was classified as fixed latency because an early experiment appeared to show the
  stalls carrying the dependency. The experiment also cut the stall on an unrelated address
  setup (finding 7).
- The `loop` kernel's failure to round-trip was recorded as a loop-carried scheduling gap
  for longer than it should have been. Bisecting the schedule one instruction at a time
  against hardware showed a single guard predicate, in a straight line of code, with
  nothing loop-carried about it.

Each of these was caught by the positive control: the vendor compiler's own output must
verify clean, and every one of them made it fail.
