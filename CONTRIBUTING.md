<div align="center">

# Contributing to basalt

Reverse engineering is a group sport. A single corrected encoding is a real contribution here.

</div>

---

## What is most useful

In rough order of value:

1. **Encodings basalt gets wrong.** If `basalt asm` produces a word that `nvdisasm` decodes to something other than what you wrote, that is the highest-value bug in the project. See [Reporting an ISA gap](#reporting-an-isa-gap).
2. **Instruction forms basalt has never seen.** The database only contains what the corpus provoked `ptxas` into emitting. If you know PTX that reaches a form we do not have, the corpus addition is usually five lines.
3. **Control-bit measurements.** Latency and scoreboard behaviour is the part most likely to be subtly wrong, and it is the part that corrupts data silently when it is.
4. **Other SKUs.** Published sm_120 measurements come from one part. Confirming or contradicting them on a different SKU is genuinely useful.

## Setup

No CUDA installation is required. The toolchain script fetches pinned redistributables, needs no administrator rights, and puts nothing on your PATH.

```bash
git clone https://github.com/sunnypatell/basalt.git
cd basalt
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python scripts/fetch_toolchain.py
export BASALT_CUDA_BIN="$PWD/third_party/cuda/13.3.1/bin"   # the script prints this line

python -m basalt.cli doctor
```

`doctor` needs no GPU. Neither does building the ISA database. Only the tests marked `gpu` need real silicon.

```bash
pytest                      # everything that runs anywhere
pytest -m "not toolchain"   # pure unit tests, no NVIDIA binaries needed
ruff check . && ruff format --check .
```

## Reporting an ISA gap

Use the **ISA gap** issue template and include, at minimum:

- the exact SASS text you expected,
- the 32-hex-character encoding basalt produced, high half first,
- what `nvdisasm -b SM120a` decodes that encoding to instead,
- your `ptxas` version, from `ptxas --version`.

That is enough to reproduce without your machine. A `.ptx` file that provokes the form is a bonus, not a requirement.

## The rule that matters

> [!IMPORTANT]
> **basalt is a clean-room project and stays that way.** Do not contribute NVIDIA source, headers, documentation text, or anything decompiled or extracted from their binaries. basalt observes the behaviour of publicly distributed executables and records it. That distinction is what keeps the project on the same footing every prior SASS assembler has stood on for over a decade, and it is only true while we keep it true.

If you are unsure whether something crosses that line, open a discussion before writing code.

## Evidence, not assertion

Claims in this repository are expected to carry the measurement that produced them. In practice:

- A new instruction form lands with the encoding it was observed at and the PTX that produced it.
- A field-layout change lands with the mutation that demonstrates it, in the shape `flip bit N, operand X changed from A to B`.
- A performance number lands with the methodology, the number of runs, and a range. Locked clocks if you have them; if not, say so and report medians.

If a test passes both before and after your change, say that in the pull request rather than letting the green check imply more than it proves.

## Commits and pull requests

Conventional commits, lowercase, past tense:

```
feat(isa): added the block-scaled QMMA.SF forms
fix(probe): stopped aliasing operand slots when a form repeats a register
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.

Keep pull requests scoped to one concern. Explain the why; the diff covers the what.

## Code style

`ruff` is the arbiter, configured in `pyproject.toml`. Beyond that: every source file carries an SPDX header, comments explain the non-obvious *why* rather than restating the code, and module docstrings explain what a reader needs to know before reading the module.

No runtime dependencies. The pipeline shells out to `ptxas` and `nvdisasm` and parses text, deliberately, so the database can be rebuilt from a pinned redistributable and nothing else.
