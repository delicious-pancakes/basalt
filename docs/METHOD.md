<img src="assets/header-method.svg" alt="basalt Method" width="100%" />


How basalt arrives at each thing it claims, and what would falsify it. Every number
in this document regenerates from a clean checkout with the commands shown.

## 1. The two oracles

basalt drives two stock NVIDIA binaries as external processes and reads their output.
It links no NVIDIA code and redistributes none.

**Ground truth.** `ptxas` compiles generated PTX to a cubin; `nvdisasm -c -hex` prints
each instruction with its 128-bit encoding. Every pair produced this way is something
the vendor compiler really emitted, so its meaning is not in question.

**Probe.** `nvdisasm -b SM120a` decodes a flat file of arbitrary 16-byte words. This is
the more useful of the two, because it answers questions about encodings `ptxas` would
never produce, which turns the instruction set from something to guess at into something
to measure.

Neither needs a GPU. The instruction database therefore rebuilds in CI on any machine,
which is the property that makes it auditable by someone who does not own the hardware.

> [!NOTE]
> Raw mode aborts the whole file on the first illegal instruction and prints nothing, so
> one bad word in a batch loses the good ones too. It does name the offset it choked on,
> which is enough to split the batch around the offender rather than falling back to one
> process per word. That distinction is roughly two orders of magnitude on a full probe.

## 2. Deriving the encoding by changing it

For a form to be useful to a checker, it is not enough to know that an encoding exists;
the fields inside it have to be located. basalt does this by mutation.

Take an encoding that assembled, flip one bit, decode the result, and classify the
difference: a bit that moves the destination register is a destination bit, a bit that
changes the mnemonic is a selector, a bit that changes nothing observable is inert, and
a bit the decoder rejects is structural.

Against `IADD R5, R5, 0x2a`:

| Role | Bits | Evidence |
| :--- | :--- | :--- |
| operand 0 | 16:23 | flip 16 gives `R4`, flip 17 gives `R7` |
| operand 1 | 24:31 | plus bit 72, which prints the source negated |
| operand 2 | 32:63 | flip 32 gives `0x2b`, flip 33 gives `0x28` |
| opcode | 2, 4, 12:15 | mnemonic changes |
| inert | 36 bits | no observable effect |
| invalid | 11 bits | decoder rejects the mutation |

Eight-bit register fields and a 32-bit immediate, by experiment.

**Choosing what to probe.** A form printed as `IADD R5, R5, 0x2a` cannot separate the
destination field from the source field, because both hold 5 and a flip in either looks
the same. The database therefore prefers a representative whose register operands are
distinct. Skipping that step produces a field map that looks fine and is wrong.

```bash
python -m basalt.cli build-isa
python -m basalt.cli isa IMAD.WIDE.U32
```

## 3. The scheduling control word

Each instruction carries 21 bits the hardware does not check, and the table below adds
up to exactly that.

| Field | Bits | Meaning |
| :--- | :--- | :--- |
| `stall` | 108:105 | Cycles before the next instruction issues |
| `yield` | 109 | Hint that the scheduler may switch warps |
| `write_barrier` | 112:110 | Scoreboard signalled on write-back, 7 for none |
| `read_barrier` | 115:113 | Scoreboard signalled once sources are consumed, 7 for none |
| `wait_mask` | 121:116 | Scoreboards that must be clear before issuing |
| `reuse` | 125:122 | Operand reuse-cache flags |

The layout validates itself against compiler output. `S2R` signals scoreboard 0 and the
`IMAD` consuming it waits on mask `0x01`; `LDCU.64` signals 1 and the dependent `STG.E`
waits on `0x02`; instructions `nvdisasm` annotates `.reuse` carry the matching reuse bit.
Every producer and consumer pair lines up. These are held as tests rather than notes.

## 4. Measuring latency

The checker is only as good as its latency model, so the model is measured rather than
assumed.

A dependent chain of the instruction under test is timed with the cycle counter, one warp
and one block so nothing overlaps. Three properties keep the result honest.

**Slope, not a timing.** Measuring one chain conflates the instruction with the fixed cost
of the clock reads and the launch. Fitting cycles against chain length across several
lengths cancels every constant exactly, whatever it happens to be, without having to know
what it was.

**The chain is counted.** `ptxas` folds arithmetic. An earlier `LOP3` chain written as an
xor followed by an and collapsed into a single lookup, and a measurement that divided by
the requested length would have reported a latency four times too large with no sign
anything was wrong. Every kernel is disassembled and its instructions counted, and the
observed count is what enters the fit.

**Results that do not look like a latency are withheld.** A clean dependent chain is
almost perfectly linear and lands on a whole cycle. A fit below R² 0.999, a non-positive
slope, or a result more than 0.2 cycles from an integer is rejected rather than published
with a caveat nobody reads.

```bash
python -m basalt.cli measure -o data/latency/your-card.json
```

### Results on an RTX 5070 Ti

70 SMs, `ptxas` V13.3.73, every fit R² ≥ 0.9998.

| Instructions | Cycles |
| :--- | ---: |
| `IMAD` `IADD3` `FFMA` `FADD` `FMUL` `LOP3` `SHF` | 4 |
| `POPC` | 18 |
| `I2FP` + `F2I` together | 24 |
| `MUFU` | 44 |
| `DADD` `DFMA` | 64 |

Three of these contradict the assumed model basalt shipped with: `DADD` was assumed 48,
`POPC` was assumed 4, and each conversion was assumed 6 against 24 for the round trip. An
assumed latency model is not a small approximation of a measured one.

**What is deliberately not claimed.** A conversion cannot feed the next link of a chain
without converting back, so `I2FP` and `F2I` cannot be separated by this method. The pair
is reported as a pair and kept out of the per-opcode model rather than halved.

**A deterministic latency is not a stall-covered latency.** `MUFU` fits a perfectly linear
44 cycles, but `ptxas` signals a scoreboard on it and the dependent instruction waits on
that scoreboard, so it stays classified as variable. What matters to the checker is how
the compiler is required to cover an instruction, not how predictable it happens to be.

## 5. Checking a program

With an encoding model and a latency model, the rules the hardware does not enforce can
be checked directly.

- **Fixed-latency results.** For a producer at `i` with latency `L` and a consumer at `j`,
  the elapsed time is `sum(stall[i..j-1])`, and it must be at least `L`.
- **Out-of-order results.** The producer must signal a scoreboard and something must wait
  on it before the consumer issues. Scoreboards are counters rather than flags, so several
  producers may share one and a single wait covers every outstanding signal, for every
  instruction downstream.
- **Operands still being read.** An instruction signalling a read barrier has not consumed
  its sources yet, so anything overwriting those registers must wait on it.
- **Operands read without a barrier.** The same hazard where the compiler covered it with
  spacing instead. The gap needed is a property of the pairing rather than a constant, so it
  is mined from what the vendor leaves, and reported only where there is enough evidence to
  say a shorter gap is wrong (finding 23).

Analysis is per basic block. Tracking a definition across a branch needs a real
control-flow graph, and inventing one from a linear listing produces confident nonsense.

The scheduler answers the same four questions in the other direction, from the same model
and, where a rule has a number in it, the same function. A checker and a scheduler that
disagree produce schedules that fail their own verifier, which has happened here and is
worth designing against rather than testing for.

```bash
python -m basalt.cli verify path/to/kernel.cubin --latencies data/latency/rtx-5070-ti.json
```

## 6. The controls

> [!IMPORTANT]
> A checker that reports "clean" unconditionally passes every test you would think to
> write for it. Both of these run in CI.

**Positive control.** The vendor compiler's own scheduling must verify clean. If basalt
flags `ptxas` output, basalt is wrong, and nothing it says about anyone else's code counts
until that is fixed. This control has already earned its place twice: it caught a rule
treating scoreboards as flags rather than counters, and a rule requiring each consumer to
carry its own wait. Together those produced eight false findings in a thirty-two
instruction kernel.

**Negative control.** Take that same clean output, shorten one stall count, and basalt
must flag exactly that instruction and nothing else. Detection matters as much as silence,
and collateral false alarms matter as much as detection.

**Cross-validation.** The ELF reader's view of the instruction stream is compared against
`nvdisasm`'s, word for word. Two independent paths to the same bytes agreeing is cheap
evidence that neither is confused.

**Agreement between models.** The measured latency model verifies real compiler output
clean. An independently measured number and the vendor's scheduling decisions agreeing is
good evidence that both are right, and it is not proof: they can also agree because the
same wrong assumption is in both.

**The hardware round trip.** Which is why this exists, and why it is the control the others
answer to. Every kernel the corpus generates is compiled, stripped of every control bit,
rescheduled from scratch, and run on the GPU beside the vendor's version of itself. The
output bytes have to match.

Nothing else here is independent of basalt's own latency model. The checker and the
scheduler both read it, so a wrong entry satisfies both at once and they agree while both
being wrong; only the silicon has no stake in the argument. Running the scheduler over
seven hand-written kernels passed seven of seven for a long time, and running it over three
hundred found forty-one that were wrong.

**Four inputs, not one.** A stale read only changes the answer when the stale value and the
fresh one differ, so a single pattern of bytes is a single chance to notice. Every kernel in
the round trip runs against eight patterns chosen to disagree with each other everywhere.
Adding the second, third and fourth immediately exposed a carry-out predicate the operand
model had been reading as a source since the beginning, which had survived every control
above it including the round trip.

**Three optimisation levels, not one.** `-O1`, `-O2` and `-O3` are three different vendor
schedules to replace rather than one, and they exercise different machinery: `-O3` unrolls a
loop into ordinary registers where `-O1` keeps its counter in uniform ones. The uniform
datapath had no coverage at all while only `-O3` was run, and two kernels were being
scheduled wrong in it. A third bug lived at `-O1` alone, in a tensor kernel whose read
barrier depended on gaps basalt had compressed. A control that runs at one level is a
control over one code generator setting.

**Every result names the code that produced it.** Each sweep prints the source tree and
commit it imported above its verdicts. That is not bookkeeping: an editable install pointing
at a stale clone elsewhere on the disk shadowed the working tree for fifty-four commits, and
because the output was only verdicts there was nothing in it to show that the numbers
described different source than the one being edited.

**Assembling back to the vendor's bytes.** Every instruction `ptxas` emits is disassembled
to text and assembled back, and the result has to be the same 128 bits. Coverage is allowed
to be partial and is reported; the count of instructions that assemble to *different* bytes
is pinned at zero, because a word that disassembles to the right text and encodes something
else is the same failure this project exists to catch.

**Costing the result.** A scheduler that reports only whether it was right is hiding the
trade it made, so the issue cycles of both schedules are compared and the ratio is pinned
from both sides. Getting slower is a regression; getting much faster without the round trip
also moving is a reason to distrust the costing rather than to celebrate it.

**Holding out the code the audit reports on.** The stall requirement is mined from what the
compiler schedules, which means a table mined from a body of code cannot fail when checked
against that same code: the tightest gap it left *is* the floor, by construction. That is not
a hypothetical. basalt passed 1,323 kernels of positive control while carrying seven model
errors, and every one surfaced the first time the checker read machine code from somewhere
else. So the requirement is mined from one set of shipped libraries and the audit reports on
another, and the split is the control rather than a tidiness.

**Alleging and emitting are different jobs.** A mined minimum is an upper bound on the
requirement, so evidence from a wider body of code may lower what the checker is willing to
call an error. It may not lower what the scheduler emits, because a schedule has to be right
rather than defensible, and the figure collapsed over consumers is the tightest gap any one
of them got. Every entry in the table carries both numbers for that reason.
