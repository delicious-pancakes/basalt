# Changelog

All notable changes to basalt are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until 1.0, the ISA database schema and the assembler's text syntax may change in
any release. The clean-room position in [`NOTICE`](NOTICE) will not.

## [Unreleased]

### Added
- Assembler: SASS text to the 128-bit instruction word, and a `basalt assemble`
  command that can read its own output back through `nvdisasm` to prove it.
  7,597 of the 8,560 corpus instructions reassemble bit-identically to what
  `ptxas` emitted, and none assembles to anything else; the second count is a
  test pinned at zero. Branch targets and unsampled operand shapes are refused
  with a reason rather than approximated.
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
- fp64 is modelled as completing out of order rather than on a fixed schedule.
  All 41 fp64 instructions in the corpus carry a write scoreboard and none goes
  without. The 64 cycle figure stays as the cost of a dependent chain, which is
  a different question from what correctness requires.

### Fixed
- The yield bit is written from the stall rather than inherited. The two are not
  independent: across the corpus `ptxas` emits a zero stall with the bit clear
  4205 times and set never, and a stall of one with it set 1123 times and clear
  never. Leaving it as found produced pairs the vendor never emits, which
  `nvdisasm` refuses and the GPU runs anyway.
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
