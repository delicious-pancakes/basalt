# Changelog

All notable changes to basalt are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until 1.0, the ISA database schema and the assembler's text syntax may change in
any release. The clean-room position in [`NOTICE`](NOTICE) will not.

## [Unreleased]

### Added
- Stage 10, the audit: the checker pointed at 2,473 sm_120 kernels NVIDIA ships
  in CUDA 13.3.1, 844 MB of device code basalt did not compile. The first run
  reported 6,593 errors in 250 nvjpeg kernels and not one was real. Seven holes
  in basalt's model were, each of them invisible on a corpus basalt generates
  itself, and after the corrections the same libraries verify with zero errors
  and 201 warnings. Finding 32.
- `scripts/mine_shipped.py`: the per-pair stall requirement mined from shipped
  kernels rather than from a generated corpus, holding out the libraries the
  audit then reports on, because a table measured on the code it is checked
  against cannot fail. 909 cubins and 24,311 kernels take the table from 340
  dependent pairings to 3,957. Where the wider corpus disagrees with the narrow
  one it lands on the number fault injection had already measured: a guard
  predicate at 13 cycles, from 229,567 observations instead of 9.
- `scripts/fetch_toolchain.py --libs` fetches those libraries and records the
  exact component version beside each, since a finding against cuBLAS is worth
  nothing without saying which build of it.
- Whether a missing scoreboard is a hazard is now measured rather than assumed.
  The vendor covers `LDG`, `LDC`, `LDL`, `S2R` and `LD` with spacing alone zero
  times across millions of dependent pairs, and `LDS` 734,837 times, so the
  first group keeps the error and the second gets a warning.

### Fixed
- `F2IP` read `F2I`'s entry through the longest-prefix rule and was treated as
  completing out of order. It is the third instruction to fall into a trap two
  comments in the same file already warned about, and the only one the corpus
  could not reach: it appears nowhere in 37,008 instructions of compiler output.
- `R2UR` was classed variable and is never scoreboarded once in 3,098 corpus
  instances. Every one of them carries the safe stall encoding, which exempted
  it from the check that would have caught it.
- Evidence may lower what basalt alleges and never what it emits. The figure
  collapsed over consumers puts `IMAD` at 2 cycles across 733,217 observations,
  true of `IMAD -> IMAD` and nothing else, and scheduling to it broke 8 kernels
  on the card. Every entry now carries the floor a scheduler must respect
  alongside the tightest gap the checker may call an error.
- A requirement may not exceed the producer's own latency, and a guard's may not
  exceed the 13 cycles measured for it. Both are upper bounds by construction.
- A mined exact pairing was reported as grounded however thin, so four
  observations claiming `IADD -> MOV` needs 20 cycles became an error against
  code the vendor ships at 5.
- The scoreboard residue is a pipeline constant. `LDG.E.64 => FFMA` mined at 461
  cycles from 3 observations, and 2 is the only figure fault injection produced.
- A scoreboard is freed after an instruction takes its own rather than before,
  so a load can no longer end up waiting on the barrier it signals.
- A definition guarded by `@P0` no longer reaches a use guarded by `@!P0`.
  Within a thread exactly one of them runs.
- A variable-latency result the vendor covers with spacing is checked against
  that distance rather than merely noted. `DSETP.GT.AND -> FSEL` is scheduled 66
  cycles apart 4,322 times with no barrier, which is fp64's measured 64 plus the
  residue, and reporting a missing scoreboard said nothing about the 66.
- The scoreboard residue no longer warns between the measured figure and the
  mined one. The wait is what makes the pair safe, so the gap beside it carries
  no requirement, and that band was 945 of the first 1,149 warnings.

### Added
- Read barriers are derived rather than copied from the schedule being replaced,
  which is what a scheduler needs to place control bits on a program nobody has
  compiled. All 334 in the corpus were characterised first: 328 sit on a
  variable-latency instruction and the rest on a store, all 299 sit on one whose
  source register is overwritten somewhere, and the barrier goes on the last
  such reader before the overwrite because it covers every earlier one. Removing
  them makes `s_tile_matmul` return the wrong answer on the card, which is the
  evidence they are load-bearing. Finding 25.
- `cp.async` is scheduled and back in the corpus, in four forms over two cache
  modifiers and three widths. The obstacle was not the copy but `DEPBAR`, which
  names its scoreboard in the operand text rather than in the control word, so
  renumbering the `LDGDEPBAR` that signals it unpaired them with nothing in the
  encoding to show it. Finding 27.
- `scripts/verify_all.py`: run every control in order and print what each one
  answered. The README quotes about a dozen numbers and reproducing the set meant
  running the commands behind them by hand in the right order, which works once
  for the person who wrote it. A step needing hardware this machine does not have
  is reported as skipped rather than quietly passing.
- `scripts/across_the_family.py`: `ptxas` compiles the corpus to byte-identical
  code, control words included, for all six targets in this family including
  `sm_121`, which is a different chip. The schedule a kernel needs is therefore a
  property of the architecture rather than of the part, which is the per-SKU
  question answered without a second card. Finding 28.
- `scripts/cross_toolchain.py`: the instruction model derived a second time from
  CUDA 13.0.3 and compared with the 13.3.1 one. 195 of 343 exemplar encodings
  differ and no operand field does, and the 13.3.1 database assembles 13.0.3's
  output at 12,351 of 12,360 with none wrong. Stage 9b. Finding 29.
- `scripts/rebuild_and_compare.py`: the two-command drift check as one command.
- `scripts/corpus_figures.py`: recompute every corpus-derived figure the
  documentation quotes, from a fresh compile, in one command. Several had already
  drifted as the corpus grew, and one had been measured from a scratch directory
  holding cubins from two different corpus versions, which produced numbers that
  were plausible and wrong. A figure nobody can regenerate is an assertion.
- `scripts/probe_kernel.py`: reschedule one kernel and print which control field
  moved, in seconds rather than the minutes the corpus runner takes, with a
  standing check for control-word shapes `ptxas` never emits.
- Plain address operands are taken apart into base and offset, as descriptors
  already were, so every shared and local address can be assembled. Two traps
  came with it and are both refused rather than guessed: a bit that swaps the
  register file is a selector and not part of the number, and `c[0x3][R0]`
  indexes by register where an offset was assumed.
- Assembler: SASS text to the 128-bit instruction word, whole cubins as well as
  single instructions, and a `basalt assemble` command that can read its own
  output back through `nvdisasm` to prove it. Assembling every corpus kernel as
  a program with its labels resolved reproduces 59,693 of 59,760 instructions
  bit-identically across four optimisation levels and none to anything else;
  the second count is a test pinned at zero.
- `scripts/assembler_coverage.py`: the command that produces the number above.
  Compiles every corpus kernel, hands the disassembly back to the assembler and
  compares the bytes, needing no GPU. It runs in CI so the published figure is
  regenerated on every push rather than asserted.
- Operand modifiers are separated from the values beside them. `-R0` is a
  register number and a negate bit that sits nowhere near it, bit 72 for `IADD`
  source 1 against bits 24:31 for the number, and the prober had already
  recorded which was which. 276 negate, 135 absolute, 104 invert and 11
  bitwise-not fields, worth 54 more instructions assembled exactly.
- A CycloneDX bill of materials is built and attested with every release, from a
  throwaway environment holding only the wheel, so it lists exactly one
  component. That is the machine-readable form of "no runtime dependencies".
- The branch target encoding, solved from real kernels rather than probed: a
  field split across bits 16..23 and 34..81, holding the distance to the
  destination from the following instruction, scaled by four. All 1,548 branches
  in the corpus decode to their label and none decodes wrongly, and a test
  re-derives it from the corpus so it cannot rot when a compiler version
  changes.
- Composite operands are taken apart into their sub-fields. A constant-bank
  reference is one field holding a bank, a base register and an offset, and the
  prober had already recorded what each bit did to the text, so decomposing it
  costs no extra probing. Schema version 2; 35 fields decomposed.
- `scripts/roundtrip_corpus.py`: reschedules every kernel the corpus generates
  from scratch and runs both versions on the GPU with identical input, comparing
  output bytes. All 314 comparable kernels come out byte-identical to the vendor
  schedule, from 246 when the control was first run. Every model
  correction in this release came out of it. Recorded as finding 10 in
  [`docs/FINDINGS.md`](docs/FINDINGS.md), with the failures named.
- Corpus-wide scheduler check in the test suite: every kernel is rescheduled and
  handed back to the verifier. Needs no GPU, so it runs in CI, and it is the
  floor rather than the ceiling; only the round trip above sees a wrong latency
  entry that satisfies checker and scheduler alike.
- Guard predicates modelled as their own hazard class. A predicate consumed as
  an instruction's guard needs 13 cycles from a fixed-latency producer where the
  same predicate read as data needs 5, because a guard is resolved before issue
  rather than at operand read. Measured on hardware by stall sweep and confirmed
  across the corpus; recorded as finding 9 in [`docs/FINDINGS.md`](docs/FINDINGS.md).
  The checker, the scheduler and the stall miner each keep guard and data
  evidence in separate keyspaces, including the per-producer fallback.
- Per-opcode minimum stall for a producer that also signals a scoreboard, mined
  from the compiler the same way the pair-wise requirements are. A wait covers
  the long part of a variable-latency result and not the whole of it: `ptxas`
  never schedules a `DADD` below 2 cycles however the consumer waits, and one
  cycle less computes a different answer.
- Scoreboard waits propagated along control-flow edges to a fixed point, so a
  value produced in one block and consumed in another is waited on rather than
  relying on the block-local pass that could not see it.
- Toolchain layer driving `ptxas` and `nvdisasm` as external processes, with a
  fetcher for pinned CUDA redistributables so no full CUDA install is needed.
- Cubin ground-truth oracle and raw-word probe oracle, both GPU-free.
- The 128-bit instruction word with its 23-bit control section, confirmed
  against producer/consumer pairs in real compiler output.
- PTX corpus generator covering scalar arithmetic, conversions, memory spaces,
  atomics, warp-level primitives and control flow.
- Tensor corpus reaching `HMMA`, `IMMA`, `QMMA` over the FP8/FP6/FP4 types
  including mixed operand types, the block-scaled `QMMA.SF` and `OMMA.SF`
  families, sparse `IMMA.SP`, and `LDSM`/`STSM`/`MOVM`.
- Differential bit prober that infers per-bit roles by mutation.
- Generated ISA database with a schema version, coverage counts, and full
  provenance for every entry.
- ELF reader that locates `.text.<kernel>` and rewrites single instruction words
  in place, which is what makes negative controls possible.
- Hazard checker over a real control-flow graph: reaching definitions carrying
  the minimum stall along any path and whether a scoreboard was waited on along
  every path.
- Latency measurement on real silicon through the CUDA driver API, reached with
  `ctypes` alone so no toolkit, host compiler or third-party binding is needed.
- Fault injection: the required stall determined by shortening the gap until the
  answer changes, with a sensitivity control so a kernel that cannot detect a
  stale read reports nothing rather than reporting a small number.
- Per-pair stall requirements mined from the vendor compiler's own scheduling.
- Field validation: every measured operand field is written through and read back,
  so the database is shown usable rather than merely readable.
- Scheduler that assigns the control bits from the same model the checker uses,
  verified against its own checker and then against the hardware.


### Changed
- The last quantity basalt copied from the schedule it was replacing is derived.
  Inside the window a read barrier covers, the vendor's own stalls were a floor,
  which worked and meant nothing for a program the vendor never compiled. The
  floor is now the issue rate mined from every consecutive pairing in the corpus:
  `LDG` after `LDG` is 4 cycles across 1,953 observations, which is the same
  number and a derived one. Nothing in the control word is inherited any more.
- The anti-dependency requirement is mined per pairing rather than charged as a
  constant. Three cycles is what fault injection measured for `ULEA` into
  `UMOV`; the vendor leaves one or two for hundreds of other pairings, 386 times
  across the corpus, so a constant was both over-charging and unprovable.
  Evidence only ever lowers the charge, because the smallest gap `ptxas` left is
  frequently there for another reason. The checker tests the same rule through
  the same function, which closed a gap where the scheduler charged for a hazard
  the checker never looked for. Issue cost 1.06x to 1.05x.
- The yield bit follows a rule fitted to 37,008 instructions of vendor output,
  93.7% agreement against the 73.2% the previous guess managed. Inverting 680 of
  them in the vendor's own schedules changed no result on the card, so the field
  is a hint rather than a correctness input, which is now measured rather than
  repeated. Finding 26.
- Shared and local `ld`/`st` corpus kernels used the global pointer as an address
  in the wrong space: they compiled, faulted, and had been excluded from the round
  trip for as long as they existed. They now address a slot they own, as do the
  shared atomics, the shared reduction and every `ldmatrix`, so the round trip's
  exclusions fell from 19 to 2. Both remaining ones read the clock or the grid id.
- fp64 is modelled as completing out of order rather than on a fixed schedule.
  All 219 fp64 instructions in the corpus carry a write scoreboard and none goes
  without. The 64 cycle figure stays as the cost of a dependent chain, which is
  a different question from what correctness requires.

### Fixed
- `python scripts/fetch_toolchain.py --list` divided a string by a float and had
  never run. The redistributable manifest quotes the size as a string.
- Removed a repair loop that reallocated scoreboards away from any instruction
  waiting on the number it signals. `ptxas` emits that shape 251 times in 37,008
  instructions, so the rule was invented to avoid a hazard that is not one, and
  it was not free: forbidding a number pushes the allocator onto a fresh
  scoreboard, and `s_tile_matmul` was taking 57 where it needs 20. Removing it
  changes no answer, 439 of 439 at every level. Finding 24 records what the
  symptom did lead to, which was two real causes.
- A memory access width describes the data, not the address. `STS.128 [R0], R8`
  moves four registers to one 32-bit shared address, and widening the address
  invented a dependency on `R1` through `R3` that made the vendor's own schedule
  read as broken. Caught by the positive control the moment the corpus grew
  kernels that use it.
- A producer is credited when a wait covers it rather than when one was written
  for it. A scoreboard is a counter, so a wait placed for one producer drains
  every producer sharing the number, and crediting only the intended one dropped
  barriers that downstream code was leaning on.
- The scheduler no longer tightens the gaps a read barrier depends on. `ptxas`
  puts one barrier on the last of a run of loads and lets in-order issue carry
  the earlier ones, so compressing the run makes it fire while they are still
  reading. `k_mma_m16n8k32_s4_s4_s32` at `-O1` overwrote `R4` under four loads
  that had not finished reading it: no fault, wrong answer, found by the round
  trip and now held by a test that fails when the rule is narrowed.
- A wait that services a read barrier is kept on the instruction that does the
  waiting rather than the one that sets the barrier, which is where it was being
  folded in and where it protects nothing.
- Every sweep prints the source tree and commit it imported. An editable install
  pointing at a stale clone shadowed the working tree for fifty-four commits and
  the test suite ran against the old source without anything saying so; `src` is
  now pinned on the pytest path as well.
- The yield bit is written from the stall rather than inherited. The two are not
  independent: across the corpus `ptxas` emits a zero stall with the bit clear
  17,999 times and set never, and a stall of one with it set 6,192 times and
  clear twice. Leaving it as found produced pairs the vendor emits rarely or not
  at all, which `nvdisasm` refuses and the GPU runs anyway.
- Two checks that reschedule a kernel and hand it back to the verifier now
  compare the instruction count first. A program `nvdisasm` will not read comes
  back empty, an empty program has no hazards, and both had been reporting clean
  on nothing for as long as the encoding bug existed.
- A call waits for everything still outstanding. Control leaves for code this
  analysis has not read and the callee may use any register, so nothing may be
  in flight when it issues. Indirect branches are treated the same way, since
  their destination is computed. Fixed both 4-bit integer MMA kernels.
- The scheduler no longer leans on a wait carried by a predicated instruction.
  That instruction may not execute, so a later consumer relying on its wait can
  read a result that never landed. `MUFU.EX2`, `SQRT` and `RSQ` feeding a store
  through a predicated `FMUL` were wrong every time until the store carried its
  own wait. The checker deliberately keeps the permissive rule, because `ptxas`
  does lean on predicated waits and its output runs.
- `MOVM` is modelled as completing out of order, like the matrix load beside it.
- `IMAD.WIDE` writes a register pair, and its addend is one. It computes a
  64-bit `a * b + c` from 32-bit factors, with nothing in the mnemonic to say
  which operands are wide; the `.U32` some forms carry describes the factors
  rather than the result. Both high halves were invisible.
- The round trip runs the vendor a second time, after basalt has had the card,
  and excludes any kernel that no longer agrees with itself. A kernel reading
  uninitialised shared memory is stable until something else has used the GPU,
  so its first result is not ground truth; every `LDSM` and `MOVMATRIX` kernel
  read as a basalt failure until this check existed.
- Fixed-latency pair requirements are keyed on the producer's full mnemonic, not
  its bare opcode. `IMAD.WIDE.U32.X` into `IADD` is scheduled at 3 cycles and
  plain `IMAD` into `IADD` at 5; collapsing them applied one instruction's
  requirement to another. An exact pairing now wins however few times it was
  seen, since on an exact pairing a thin sample is the only evidence there is.
- `BREV` is modelled as completing out of order. It carries a scoreboard in two
  of the three dependent instances in the corpus and the third is covered by a
  wait on a later instruction from the same unit.
- An instruction that writes a predicate and a register is read as writing both.
  `SHFL.IDX PT, R9, ...` and `ATOMG ... PT, R7, ...` return a value as well as a
  predicate, and reading only the predicate meant nothing scoreboarded that value
  and nothing waited for it. Fixed every atomic in the corpus.
- A scoreboard signalled by one instruction now covers earlier results the same
  unit still owes, since a unit returns results in the order it was given work.
  `ptxas` scoreboards the second of two consecutive `SHFL.IDX` and waits only on
  that one, which the checker had been reporting as a hazard in vendor output.
- A predicated write no longer kills the previous definition of its register.
  `@!P0 FMUL R7, R7, c` leaves R7 holding whatever produced it wherever the guard
  is false, so both reach any later reader and both have to be covered.
- The `loop` and `double` kernels now round-trip through hardware, taking the
  scheduler to seven of seven. Both had been carried as expected failures
  blaming missing passes, a loop-carried dependence and an unmodelled
  stall-plus-scoreboard combination. Neither diagnosis was right. Bisecting each
  schedule one instruction at a time against the GPU put the first on a single
  guard predicate in straight-line code and the second on a wrong latency class.
- Finding 7 in [`docs/FINDINGS.md`](docs/FINDINGS.md) concluded that stalls
  rather than scoreboards carry an fp64 dependency. The experiment behind it also
  cut the stall on an unrelated address setup, which breaks the kernel by itself.
  The entry now records both the correction and why it went unnoticed.

### Found
- A read barrier is set exactly where an operand read outlives its register, and
  covers every earlier late reader before the overwrite. All 299 in 37,008
  instructions of vendor output fit that rule, and all 318 in-block overwrites
  left bare are accounted for by three mechanisms that add up exactly. Finding 25.
- The yield bit does not gate correctness on sm_120. 680 inversions across fp64,
  transcendental, tensor, loop, barrier and shared-atomic kernels, and the card
  computed the vendor's answer every time. Finding 26.
- A scoreboard can be named in an operand rather than in the control word.
  `DEPBAR.LE SB0, 0x0` is the only instruction that does it, and it is how
  `cp.async` synchronises. Anything that rewrites control words has to know.
  Finding 27.
- A read barrier covers more reads than its own instruction's. `ptxas` puts one
  on the last of a run of loads and relies on the gaps it chose to carry the
  rest, which nothing in the encoding records. Recorded as finding 13, along
  with the bisection that separates "this is a stall" from "this is a barrier"
  in a single run.
- A modifier is one character of text and one bit of encoding, and the bit is
  nowhere near the operand it belongs to: negating `IADD` source 1 is bit 72
  while its register number is bits 24:31. Finding 14.
- A stall count of 0 is a safe long-wait encoding rather than zero cycles, which
  is why `ptxas -O0` emits a zeroed control word and still computes correctly.
- Understalling corrupts silently and deterministically; basalt's static verdict
  matches the hardware for every encodable stall value.
- Measured latency, and tensor-core throughput, for sm_120 on an RTX 5070 Ti.
  See `docs/FINDINGS.md`.
- A predicate used as an instruction's guard needs 13 cycles where the same
  predicate read as data needs 5. The discriminating case is `LOP3 -> MOV`,
  which appears in the corpus both ways with both numbers, so the cost belongs
  to issue rather than to the predicate file or the opcode pair.
- A waited-on scoreboard does not settle a dependency on its own: the producer
  still owes a gap to its consumer, 2 cycles for `DADD`, and one cycle less
  changes what the GPU computes.
- The yield bit is not independent of the stall count. `ptxas` emits a zero
  stall with the bit clear 4,205 times and set never, and a stall of one with it
  set 1,123 times and clear never. The unseen combinations are words `nvdisasm`
  refuses and the GPU runs anyway.
- Where a branch keeps its destination: a field split across bits 16..23 and
  34..81, holding the distance from the following instruction scaled by four.
  All 1,548 branches in the corpus decode to their label.
- `EXIT`, `RET`, `CALL` and `BAR` never take the zero-stall encoding, in 0 of
  329, 0 of 5, 0 of 5 and 0 of 3 instances, while `BRA` takes it 329 times.
