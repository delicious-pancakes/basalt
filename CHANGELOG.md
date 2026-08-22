# Changelog

All notable changes to basalt are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until 1.0, the ISA database schema and the assembler's text syntax may change in
any release. The clean-room position in [`NOTICE`](NOTICE) will not.

## [Unreleased]

### Added
- Guard predicates modelled as their own hazard class. A predicate consumed as
  an instruction's guard needs 13 cycles from a fixed-latency producer where the
  same predicate read as data needs 5, because a guard is resolved before issue
  rather than at operand read. Measured on hardware by stall sweep and confirmed
  across the corpus; recorded as finding 9 in [`docs/FINDINGS.md`](docs/FINDINGS.md).
  The checker, the scheduler and the stall miner each keep guard and data
  evidence in separate keyspaces, including the per-producer fallback.
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


### Fixed
- The `loop` kernel now round-trips through hardware, taking the scheduler to
  six of seven. It had been recorded as a loop-carried scheduling gap; bisecting
  the schedule one instruction at a time against the GPU showed a single guard
  predicate in straight-line code, and nothing loop-carried about it.

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
