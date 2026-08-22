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
