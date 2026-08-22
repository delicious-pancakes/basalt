# Changelog

All notable changes to basalt are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until 1.0, the ISA database schema and the assembler's text syntax may change in
any release. The clean-room position in [`NOTICE`](NOTICE) will not.

## [Unreleased]

### Added
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

### Found
- A stall count of 0 is a safe long-wait encoding rather than zero cycles, which
  is why `ptxas -O0` emits a zeroed control word and still computes correctly.
- Understalling corrupts silently and deterministically; basalt's static verdict
  matches the hardware for every encodable stall value.
- Measured latency, and tensor-core throughput, for sm_120 on an RTX 5070 Ti.
  See `docs/FINDINGS.md`.
