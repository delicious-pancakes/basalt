<img src="assets/header-findings.svg" alt="basalt Findings" width="100%" />


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
| Kernels compiled | 393 |
| Dependencies checked | 6,359 |
| Errors | **0** |
| Kernels with warnings | 1 |

The remaining warning is a late-read case, reported as a warning rather than an
error because basalt has no measured model of how long an operand read takes.
Finding 13 is what that gap costs and what basalt does about it.

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
| Forms checked | 270 |
| Register-bearing operand slots | 713 |
| Slots that behave as measured | **695 (97.5%)** |
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

Every one of the 50 fp64 instructions in the corpus carries a write scoreboard
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

**Across the corpus.** Every dependent pair in the 393 kernel corpus, split by how the
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

`scripts/roundtrip_corpus.py` runs it over everything the corpus generates, plus thirteen
hand-written kernels whose control flow is the point rather than their opcodes. For each of
the 330: compile with `ptxas`, discard every control bit it produced, compute new ones,
write them back, and run both versions on the card with identical input. The rescheduled
kernel has to produce the same bytes.

The thirteen exist because the generated corpus is deliberately narrow. One or two
instructions of body per kernel is right for attributing a bit to a form and wrong for
exercising a scheduler: almost nothing in it has a loop, a barrier, a nested branch, or
shared memory that is actually addressable. The hand-written ones have counted loops with
accumulators carried around the back edge, nested loops, a branch inside a loop, barriers
with traffic on both sides, predicated writes, and long unbranched dependent chains.

| | |
| :--- | ---: |
| Kernels rescheduled and run | 406 |
| Comparable (the vendor runs here, deterministically, and reproducibly) | 314 |
| **Byte-identical to the vendor schedule** | **314** |
| Wrong | 0 |

Every kernel basalt can be compared on now computes exactly what the vendor's schedule
computes, from control bits it worked out itself. The comparable count moves by one or two
from run to run, because a few kernels' reproducibility depends on what last used the card;
the match count moves with it and the failure count stays at zero.

It holds at three optimisation levels, which is three different vendor schedules to
replace rather than one. All three legs below come from a single run on a clean tree at one
commit, which the sweep prints above its own results:

| `ptxas` level | Comparable | Matching | Mismatched |
| :--- | ---: | ---: | ---: |
| `-O1` | 315 | 315 | 0 |
| `-O2` | 313 | 313 | 0 |
| `-O3` | 314 | 314 | 0 |

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

### How much of the ISA this actually covers

"Every comparable kernel" is a claim about kernels, and the useful question is how much of
the instruction set they contain between them:

| | |
| :--- | ---: |
| Opcodes in the database | 77 |
| Opcodes the round trip executes on the GPU | **74 (96%)** |

The three it never runs are `ENDCOLLECTIVE`, `R2UR` and `WARPSYNC`, and the reason is
specific rather than incidental. The harvest compiles at `-O0` as well as `-O3`, and those
three appear only at `-O0`: `shfl.sync` and `bar.sync` lower to them there and the optimiser
folds them away by `-O3`. The round trip deliberately does not run `-O0`, because that level
emits a zeroed control word, so there is no schedule to replace and nothing the comparison
would prove.

So they are in the database legitimately and are unreachable by this control by
construction. Nothing is claimed about how basalt schedules them.

`BMSK` used to be in that group for the same reason, reachable only through `bfi.b32` at
`-O0`. Written directly as `bmsk.clamp.b32` it survives `-O3`, so a kernel was added that
does, and it is now covered.

This is the number to attack. Every correction in this document came from widening what
gets run, twice from the corpus growing and once from running the same kernels against more
than one input, so the ceiling here is coverage rather than cleverness.

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

## 11. Where a branch keeps its destination

A branch cannot be assembled on its own. The field holds a distance rather than an address,
so `BRA \`(.L_x_0)` is a different 128 bits in every kernel it appears in, and an assembler
that reuses a harvested encoding emits a jump to wherever the harvested kernel jumped.

The field resisted probing. Flipping one bit at a time and reading the decoded text back,
which is how every other field here was found, reports 95 bits as moving the target, because
changing the opcode changes what the rest of the word is read as. Searching for a contiguous
run whose value matches the distance finds nothing either, at any width, under any
convention.

It falls out immediately from real kernels instead. The label table gives the destination,
the instruction gives its own address, and the word gives the bits, so a field and a
convention that agree on every sample is the encoding:

| Instruction at | Jumps to | Distance | Bits 16:23 | As signed |
| ---: | ---: | ---: | ---: | ---: |
| `0x0b0` | `0x180` | `+192` | `0x30` | `+48` |
| `0x170` | `0x0e0` | `-160` | `0xd8` | `-40` |
| `0x230` | `0x080` | `-448` | `0x90` | `-112` |
| `0x270` | `0x270` | `-16` | `0xfc` | `-4` |

Every distance is four times the signed byte at bits 16:23, and the sign continues into bits
34:81, which sit far away with ten unrelated bits between them. So:

> **target = address + 16 + 4 × signed(bits[16:23] ++ bits[34:81])**

The split is why a contiguous search finds nothing, and the scale of four is why a search
for the raw distance finds nothing either. All 354 branches in the corpus decode to their
label under this rule and none decodes wrongly.

The rule is a measurement, and a measurement written down as a constant is exactly what goes
quietly wrong when a compiler version changes, so it is re-derived from the corpus by a test
rather than trusted. With it, assembling every corpus kernel as a whole program reproduces
10,406 of 10,416 instructions bit-identically, and none to anything else.

## 12. What the correctness costs

A scheduler that reports only whether it was right is hiding the trade it made. basalt's
schedules are correct on every comparable corpus kernel, and here is what they cost:

| | Issue cycles |
| :--- | ---: |
| `ptxas -O3` | 13,571 |
| basalt | 12,168 |
| | **0.87x** |

Slower on 34 of the 406 kernels and cheaper on the rest, with every comparable kernel still
byte-identical on the GPU at all three optimisation levels.

Cheaper than the vendor is believable rather than suspicious, for a specific reason: basalt
schedules every dependency at the tightest gap `ptxas` was ever observed to leave for that
exact pairing, and `ptxas` does not always schedule at its own minimum. It is balancing
register pressure and memory alongside issue latency; this is optimising one number.

It was not believed on sight. The first time the ratio went under 1.0 the hardware round
trip broke, on an `IMNMX` whose destination the operand model could not see, and the number
only stood once that was fixed and every kernel round-tripped again.

Two decisions still cost cycles, and both were taken deliberately.

**The safe stall encoding where a value leaves a block.** A definition consumed in another
block is covered by putting a zero stall on the block's last instruction, which waits for
outstanding results as well as elapsed cycles and costs about 37 cycles.

It used to be placed at every boundary regardless, which was unconditionally correct and
most of the gap: 732 of them across the corpus, against 74 now. Restricting it to blocks
that actually have something live out took the ratio from 1.39x to 1.29x. Live-out is
computed as the ordinary backwards fixed point rather than guessed, because trading a known
cost for an unknown correctness risk is the wrong way round.

**Stall the assignment placed and nothing needs.** The fixed point only ever adds: it finds
a consumer that is short and spends cycles in the window before it, and a cycle spent for
one pair also separates every other pair spanning that point. So a later pair can be
satisfied by stall placed for an earlier one, leaving the earlier placement larger than
anything requires. Walking that back, one cycle at a time, judged by the same requirement
function that placed them, took 1.29x to 0.87x. `LDC` alone was half the excess before it,
almost all overshoot rather than requirement.

**Not leaning on a wait a predicated instruction carries.** `ptxas` does lean on them and
its output runs; basalt emits its own wait instead, because relying on one was measurably
wrong for `MUFU` feeding a store through a predicated `FMUL` (finding 10). One extra wait
per occurrence.

### Costing this is easy to get wrong

The first attempt at this measurement reported basalt as nearly four times **faster**,
which would have been a lovely thing to write down and completely false. Everything after
the first `EXIT` is padding the assembler emits to fill a cache line and never issues, and
`ptxas` leaves it at a zero stall. Counting that padding at 37 cycles each charged the
vendor several hundred phantom cycles per kernel.

The number is a cost model, not a measurement. It counts what the control bits ask the
scheduler to wait, which is the part basalt decides, and says nothing about memory latency
or occupancy. It is pinned in the test suite from both sides: getting slower is a
regression, and getting much faster without the hardware round trip also moving is a
reason to distrust the costing rather than to celebrate.

## 13. A read barrier covers more reads than its own instruction's

A read barrier means "signal once my operands have been read". What it protects is less
obvious: `ptxas` puts one on the *last* of a run of loads and lets in-order issue plus the
gaps it chose carry the rest. By the time the fourth load has read the address register, the
first three have too, so one barrier covers four reads. Compress those gaps and the
guarantee disappears with them, silently, because nothing in the encoding records that the
barrier was ever standing in for its neighbours.

`k_mma_m16n8k32_s4_s4_s32` at `-O1` is the case that exposed it:

```
      LDG.E R7, desc[UR4][R4.64]          stall 4
      LDG.E R2, desc[UR4][R4.64+0x4]      stall 4
      LDG.E R0, desc[UR4][R4.64+0x40]     stall 4
      LDG.E R3, desc[UR4][R4.64+0x80]     stall 2   read_barrier 0
      MOV   R4, 0x90                                wait 0x01
```

Four loads take their address from `R4`, and the `MOV` that overwrites `R4` waits for the
barrier. basalt kept the barrier, kept the wait, and issued the loads one cycle apart
instead of four. The barrier fired while the earlier loads were still reading, `R4` was
overwritten under them, and they loaded from the wrong address. No fault, wrong answer,
full speed. It is the exact failure this repository exists to catch, produced by basalt.

**How the cause was isolated.** The first attempt was to start from the vendor's cubin and
apply basalt's control words to a growing prefix, which pointed at the fourth load. That
answer was wrong, and wrong in an instructive way: scoreboard numbers are global, so a cubin
holding basalt's write barriers and the vendor's wait masks is broken by construction and
the bisection was measuring its own hybrid. The experiment that works keeps basalt's
schedule whole and changes only stalls, which is always safe:

| Cubin | Matches the vendor |
| :--- | :---: |
| basalt's schedule | no |
| basalt's schedule, every stall raised to the vendor's | **yes** |

That separates the two hypotheses in one run. The bug is a stall, not a barrier, and a
second bisection over which stalls have to be raised lands on the run of loads.

**The rule basalt adopted.** It has no measured model of how long an operand read takes, so
it cannot compute a safe gap here. What it can do is decline to make the window tighter than
`ptxas` made it, and that is what it does: inside the window a read barrier covers, the
vendor's stalls are a floor.

The window is bounded by the previous read barrier and by the enclosing basic block, and
the block bound is not a convenience. `s_loop_double` at `-O1` has a read barrier on a
`DFMA` inside a loop body, where the thing it guards against is the *next iteration*
overwriting the operands of this one. The instructions above the label are preamble that
runs once, and pinning their stalls would cost cycles to protect nothing. More generally,
control can arrive at a branch target from anywhere, so whatever gap the fall-through path
happened to have is not a guarantee `ptxas` is relying on either.

**What the window actually needs, and what the rule costs.** The rule above is conservative
by construction, and it is worth knowing by how much. Holding the kernel at the vendor's
schedule and sweeping only the stall on the run of four loads:

| Stall on each load | Result |
| ---: | :--- |
| 0 | correct, and 0 is the long-wait encoding rather than a short gap |
| **1** | **wrong** |
| 2 | correct |
| 3 to 8 | correct |
| *vendor leaves 4* | |

Reproducible across repeated runs, and deterministic rather than a race. Two things follow.
The hazard is real: one cycle is not enough and the answer is wrong every time, so the
window is not an artifact of the harness. And the true requirement here is 2 where `ptxas`
leaves 4 and basalt therefore also leaves 4, which is two cycles per load spent to avoid
guessing.

That is deliberately not turned into a model. One pattern of four `LDG.E` in one kernel is a
data point, not a latency, and this is exactly the shape of evidence that produced the wrong
answer in finding 7. The rule stays "do not make the window tighter than the vendor made
it", and the number above records what it costs rather than justifying a shortcut.

## 14. A modifier is one bit, and it is nowhere near its operand

`IADD R5, R4, -R0` and `IADD R5, R4, R0` differ by one character of text and one bit of
encoding, and that bit is not in the operand's field:

| Operand | Register number | Negate |
| :--- | :--- | :--- |
| source 1 | bits 24:31 | **bit 72** |
| source 2 | bits 32:39 | **bit 63** |

The prober groups both into the same slot, because flipping either one changes that
operand's text. That grouping is correct and too coarse to write through: an assembler
handed `R5` against a form harvested as `-R0` sees a nine-bit field, no way to say which
part is the sign, and refuses.

Telling them apart costs nothing extra. The prober already records the operand text with
each bit clear and set, so a bit whose flip adds or removes a leading `-` is the negate, and
a bit whose flip changes `R4` to `R5` is the value. Across the database that separates 276
negate fields, 135 absolute, 104 invert and 11 bitwise-not from the values beside them, none
of them guessed.

Two details matter more than they look.

**The polarity is read off the form, not assumed.** Whether bit 72 set means negated is not
something to take on faith from the order the prober happened to record a pair in. The
reference text says whether its own encoding is negated, so the bit that has to change is
simply the opposite of whatever the reference holds.

**An unreadable bit cancels the split.** A bracket operand can lose a bit and lose only the
part it belonged to; the offset stays writable when a bank bit is unattributed. A value
cannot, because its bits are one number: writing a register number into the readable
fraction of a field encodes a different register, in silence. So if any bit in a field could
not be read, the field goes back to being whole and the assembler refuses it.

The effect on the corpus, at `ptxas -O3`, is 54 more instructions assembled bit-identically
and 54 fewer refusals:

| | Exact | Refused | Wrong |
| :--- | ---: | ---: | ---: |
| Before | 8,374 | 210 | 0 |
| After | **8,428** | 156 | **0** |

## 15. A mnemonic that depends on its operand's value hides the operand

Differential probing rests on one assumption: change a bit, and whatever moves in the
printed text is what that bit controls. `IMAD.SHL.U32` breaks it.

That mnemonic is not a separate instruction. It is what the disassembler calls `IMAD` when
the multiplier is a power of two, because then the multiply is a shift. Writing values
straight into the immediate field of one shows it plainly:

| Immediate written | Decoded as |
| :--- | :--- |
| `0x10` | `IMAD.SHL.U32 R22, R2, 0x10, RZ` |
| `0x20` | `IMAD.SHL.U32 R22, R2, 0x20, RZ` |
| `0x10000000` | `IMAD.SHL.U32 R22, R2, 0x10000000, RZ` |
| `0x30` | **`IMAD.U32`** `R22, R2, 0x30, RZ` |
| `0x5` | **`IMAD.U32`** `R22, R2, 0x5, RZ` |

Same bits, same field, different name, and the name depends on the value. So every single
flip of that immediate changes the suffix, the probe records all thirty-two bits as suffix
bits rather than operand bits, and the operand ends up with no field at all:

| Form | Bits attributed to the immediate |
| :--- | ---: |
| `IMAD R8, R2, 0x5, RZ` | 30 |
| `IMAD.U32 R12, R11, 0x10000, RZ` | 31 |
| `IMAD.SHL.U32 R22, R2, 0x10, RZ` | **0** |

The assembler then refused every `IMAD.SHL.U32` in the corpus, correctly, for want of
anywhere to put its operand. It was the largest single group of refusals there was.

**Recovering it without assuming it.** The tempting fix is to copy the field from a sibling
form, and that is exactly the kind of reasoning this repository refuses. What is done
instead: a bit that changed a suffix *and* moved exactly one operand is a candidate, and a
candidate is only accepted if writing several distinct values through the candidate bits
reproduces those values exactly and leaves every other operand untouched. A bit that really
does control a modifier fails that check and stays where it was. Only immediates are
recovered this way, because a register that moves a suffix is a genuinely different encoding
rather than a rendering of the same one.

| | Exact | Refused | Wrong |
| :--- | ---: | ---: | ---: |
| Before finding 14 | 8,374 | 210 | 0 |
| After finding 14 | 8,428 | 156 | 0 |
| After this | 8,479 | 105 | 0 |
| After signed immediates | 8,491 | 93 | 0 |
| After merging the candidates | **8,514** | 70 | **0** |

The last two rows are the same idea applied twice more. A minus on a *number* is part of
the number rather than a bit somewhere else in the word, and treating `-0x1` like `-R0`
refused every subtract-by-add in the corpus. And the recovery above only worked where the
hidden bits were the whole field: in `IMAD R8, R2, 0x5, RZ` they are bits 32 and 34, exactly
the two that are set, because clearing either leaves a power of two. Two bits of a 32-bit
field cannot reproduce a written value on their own, so the candidates have to be merged
with the bits already attributed to that operand before being checked.

## 16. When basalt says a schedule is unsafe, is it?

The positive control says basalt stays quiet on correct code. The negative control says it
complains about one deliberately broken instruction. Neither asks the question a user
actually has, which is whether a hazard basalt reports is a hazard the silicon agrees with.

`scripts/agreement_sweep.py` asks it across the corpus. Take the vendor's own working
schedule, shorten one stall on a real dependency, and collect two independent verdicts: what
basalt says statically, and what the GPU computes against four input patterns.

| | GPU agrees with the reference | GPU computes something else |
| :--- | ---: | ---: |
| **basalt: clean** | agreed safe | **missed** |
| **basalt: hazard** | over-strict | agreed broken |

The top-right cell is the one that matters. A miss is a schedule basalt called safe that
computes a wrong answer, which is the entire failure this repository exists to prevent.

| | |
| :--- | ---: |
| Kernels, one dependency shortened in each | 209 |
| Agreed broken | 74 |
| Over-strict | 84 |
| **Missed** | **0** |
| Unstable, excluded | 51 |

Which kernels land in the excluded column moves between runs, because several read memory
this harness never initialises and are therefore stable only until something else has used
the card. The share is larger than it was: the corpus now covers immediate-source and fp64
arithmetic, which read more uninitialised input than the register forms did. The missed
count does not move.

**Kernels with a loop are left out of this sweep, and not for tidiness.** This is the one
tool that breaks a kernel on purpose, and a loop keeps its trip count in a register:
shorten the stall in front of that register and the bound becomes whatever was stale there,
so the kernel never returns. `s_nested_loops` does exactly that. On a card that is also
driving a display, a kernel that never returns is a driver reset and a black screen, which
is not a price worth paying for seven more rows. The round trip still covers every one of
them, because it breaks nothing: it asks whether basalt's schedule computes what the
vendor's schedule computes, and both terminate.

**It did not start at zero.** The first run of this sweep reported **34 misses**, all of the
same shape: `IADD -> STG` shortened from 5 cycles to 1, computing a different answer, with
basalt reporting a warning rather than an error. The cause was that severity was decided by
the *producer's* generic latency confidence rather than by the evidence behind the
requirement that was actually used. `IADD` has no measured latency entry, so a shortfall
against a number mined from 44 of the vendor's own schedules came out as a suspicion.
Severity now follows the requirement's own provenance: measured or mined from at least three
vendor schedules is an error, an assumed producer latency is still a warning.

**What that cost.** Precision. Before the change, 57 shortened dependencies were called clean
and the GPU tolerated them; after it, they are flagged. Those became over-strict verdicts
rather than misses, which is the right direction on an architecture with no interlock, and
it is a real cost rather than a free win.

**Over-strict is not the same as wrong.** A schedule can be tighter than anything the vendor
emits and still return the right answer, because a stale read only changes the result when
the stale value and the fresh one differ. Running four patterns instead of one moves
verdicts out of over-strict and into agreed broken, which are cases where basalt was right
and a single pattern had not been enough to show it. The rest are unproven either way, and
they are counted separately rather than folded into an accuracy figure.

## 17. Going looking for a wrong word, rather than waiting for one

"Nothing assembles to the wrong bytes" was measured against instructions `ptxas` chose to
emit. That is the right control and it is a narrow one: the failure this repository exists
to catch does not need the vendor's cooperation, and a corpus can only ever ask about text
the compiler happened to produce.

`scripts/fuzz_assembler.py` goes looking instead. It mutates the operand text of every form
in the database, assembles the result, and hands it straight back to the decoder, holding
one property:

> assemble(text) must disassemble to text

The interesting part is telling a real defect from a difference in spelling, because most
mismatches are the second. `P7` and `PT` are one register. `URZ` prints as `UR63`. A
negative immediate prints as its two's complement. A discarded result is not printed at all,
so `VOTE.ANY RZ, PT, P0` reads back as `VOTE.ANY PT, P0`. And a suffix can follow an operand's
value, which is finding 15 again. None of those are wrong words, and the test that separates
them from one is whether what came back assembles to the *same* word.

**It found four defects in the first thirty seconds**, none of which the corpus had ever
reached:

| What broke | Why |
| :--- | :--- |
| `PLOP3.LUT`'s predicate slot turned `P1` into `UP0` | five bits holding a uniform-register flag, a number and a negate; splitting the negate off left four bits that are not a number |
| `IADD R4, R0, -0x1` refused | a minus on a *number* is part of the number, not a bit elsewhere in the word |
| an immediate too wide for its field was trimmed | the value was masked at the call site before anything could refuse it |
| `DFMA`/`FSEL` took an integer for a float field | splitting a modifier out of the field bypassed the check on what the field holds |

The first is the one that matters: it wrote a word naming a different register, which is
exactly the failure mode being hunted. It survived the corpus because the corpus only ever
puts `PT` in that slot.

After the fixes, at 120 mutations per form across five seeds:

| | |
| :--- | ---: |
| Mutations | 219,000 |
| Assembled and round-tripped | all of them |
| Same word, printed differently | counted separately |
| **Wrong** | **0** |

The seed is fixed so a failure reproduces exactly, and three seeds run in CI on every push.

## 18. The sign of a float immediate is part of the number

`FSEL R5, R0, -2.875, P0` and `FSEL R5, R0, 2.875, P0` differ by one bit, and basalt wrote
the wrong one:

```
vendor  000fe20000000000c038000000057808
basalt  000fe200000000004038000000057808
                        ^ bit 63
```

The cause is in the sub-field classifier from finding 14. That splits a modifier out of an
operand field by watching what a bit does to the printed text, and a bit whose flip adds a
leading `-` is the negate. For `-R0` that is exactly right: the sign of a register lives in
a bit of its own, nowhere near the register number. For `-2.875` it is exactly wrong. The
sign of a float is IEEE bit 31, inside the value, and calling it a modifier left the value
31 bits wide. The assembler then wrote 31 bits of a 32-bit float and dropped the sign.

The rule is one line: a leading `-` on a *literal* belongs to the literal. It already held
for integers, where `-0x1` is two's complement in its own field, and the guard simply never
covered floats because no corpus kernel had produced a negative float immediate.

**What surfaced it is the point.** No reasoning found this. The corpus was widened to cover
immediate-source arithmetic, which had never been harvested at all, and the count of words
that assemble to something other than the vendor's bytes went from 0 to 2. That number is
pinned at zero precisely so that a change like this cannot land quietly.

## 19. A register wearing a modifier is not a literal

One mnemonic covers several encodings, and the database holds one entry per operand *shape*
so it can tell them apart. The shape is what the operands are, ignoring their values:
`FADD Rd, Ra, Rb` and `FADD Rd, Ra, imm` differ in bits outside every operand field, so a
form harvested as one describes the other only by accident.

`|R0|` was read as a literal. It is not; it is a register carrying an absolute-value bit,
the same way `-R0` carries a negate. So these two collapsed into one bucket:

| Text | Read as | Actually |
| :--- | :--- | :--- |
| `FADD R4, -RZ, \|R0\|` | reg, reg, immediate | reg, reg, **reg** with a modifier |
| `FADD R7, R2, -24` | reg, reg, immediate | reg, reg, immediate |

Probing is one nvdisasm run per shape, so a bucket gets one representative. Whichever of the
two arrived first won it, and the other had **no encoding in the database at all** and had
to be refused. The same collision hid `c[0x3][R5]` behind `c[0x0][0x380]`: a register index
and a literal index behind identical brackets.

Fixing it is a matter of stripping a modifier before asking what an operand is, in both the
harvester and the assembler, which had disagreed with each other about `|R0|` as well. On
the same corpus:

| | Exact | Refused | Wrong |
| :--- | ---: | ---: | ---: |
| Before | 9,837 | 17 | **2** |
| After | **9,846** | 10 | **0** |

The ten that remain are bits the prober could not attribute to anything, spread across five
`RET.REL.NODEC`, three `LDC` base fields, one `FADD` and one `IADD3`. Those refuse rather
than guess, which is the designed answer.

## 20. Kernels that never compiled, and the count that hid them

`ptxas` rejecting a snippet is an ordinary result here. The corpus is deliberately broad and
tries forms the architecture may not have, so a rejection is recorded as a negative rather
than raised, and the harvest prints how many there were and carries on. That is the right
design, and it hid three separate bugs for as long as they had existed:

| Kernels | What was wrong | What it cost |
| :--- | :--- | :--- |
| all 22 half precision | `ld.global.f16`, a load type PTX does not have | `HADD2`, `HMUL2`, `HFMA2`, `HMNMX2` and the f16 conversions, absent from the database, the latency model and the mined stall table alike |
| `popc.b64`, `clz.b64` | a 64-bit destination for a count that is 32 bits wide | both 64-bit forms |
| `mma.m16n8k16.f16.f16.f16` | `.f16` accumulate given `.f32` registers, when it packs two halves into a `.b32` | the only f16-accumulating tensor form |

The half-precision case is the one worth dwelling on. The type table had carried `b16` as
f16's container since it was written, in a column that nothing read; `_load` interpolated the
arithmetic type straight into the instruction instead. So the corpus claimed a whole
precision class and covered none of it.

**A corpus bug and a form the architecture does not have look identical from outside.** Both
are one more rejected snippet. The difference is in the reason, which was being recorded and
never read. So the reasons are now sorted: a parse error, a type mismatch or an arguments
mismatch means the PTX is wrong, and anything else is a genuine negative. The harvest counts
the first kind separately and a test fails if there are any.

That test found `popc`, `clz` and `mma` on its first run, none of which had anything to do
with the half-precision bug that prompted it.

| | Mnemonics | Opcodes | Instructions reproduced |
| :--- | ---: | ---: | ---: |
| Before | 276 | 77 | 9,846 of 9,856 |
| After | **295** | **81** | **10,406 of 10,416** |

## 21. What is deliberately not claimed

Stated so the boundary of the evidence is visible.

- **`I2FP` and `F2I` cannot be separated by timing.** A conversion cannot feed the next link
  of a dependent chain without converting back, so the chain always contains one of each. The
  24-cycle figure is the pair, and it is recorded in a separate `composite` section rather
  than halved and presented as a per-instruction latency. Fault injection cannot separate them
  either, because the round trip is idempotent and the chain reaches a fixed point.
- **Only one card has been measured.** Everything here is a Gigabyte GeForce RTX 5070 Ti
  EAGLE OC, 70 SMs, 2542 MHz boost, named exactly because "a 5070 Ti" is not enough to
  reproduce a run. The factory overclock does not move the numbers, which are in cycles
  rather than nanoseconds, but the SM count plausibly could. Whether these figures
  hold across sm_120 parts with different SM counts is exactly the sort of thing that should
  not be assumed, and basalt records the part alongside every measurement so a second card can
  be compared rather than merged.
- **The scoreboard residual is not checked across a block boundary.** A waited-on
  scoreboard still leaves a gap the producer has to cover (finding 7), and that gap is
  mined one block at a time because a distance that spans a branch depends on which path
  was taken. So a definition that reaches its consumer through an edge is exempted from
  that one rule rather than judged against evidence collected somewhere looser. Reaching
  definitions carry a `crossed` flag for exactly this, and every other rule still applies
  to them. It is a gap in the checker's coverage, not a wrong answer, and the round trip
  covers the same ground from the other side.
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

## 22. Corrections made along the way

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
- `DSETP -> SEL` was required to be six cycles apart on the strength of three observations.
  Widening the corpus to cover immediate-source arithmetic produced kernels where the
  vendor scheduled it at two, and the mined floor came down to match. Re-mining changed
  nothing about how many dependencies are checked, 6,105 either way; it turned four false
  errors into none.
- `F2FP` was reading `F2F`'s entry, the conversion pipe, which signals a scoreboard and
  completes out of order. `ptxas` emits `F2FP` with no scoreboard anywhere and reads the
  packed result five cycles later, so it cannot be. It is on the fixed pipeline, the same
  correction `I2FP` and `VOTEU` needed before it, and the half-precision kernels are what
  finally reached it.
- `HMNMX2` had no latency entry at all, while `HADD2`, `HMUL2` and `HFMA2` beside it did.
  Nothing had noticed because no kernel producing one ever compiled.

Each of these was caught by the positive control: the vendor compiler's own output must
verify clean, and every one of them made it fail.
