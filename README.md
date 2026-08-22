<div align="center">

<img src="./docs/assets/social-preview.svg" alt="basalt: the correctness layer for NVIDIA Blackwell GPU machine code. sm_120 has no hardware interlock, so one wrong stall count makes the GPU read a stale register and return a wrong answer silently." width="880" />

<br/>

<img alt="Architecture" src="https://img.shields.io/badge/arch-sm__120%20%7C%20sm__120a-76B900?style=flat-square&labelColor=0d1117">
<a href="https://github.com/sunnypatell/basalt/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/sunnypatell/basalt/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=ci&labelColor=0d1117"></a>
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-3178C6?style=flat-square&labelColor=0d1117"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=0d1117">
<img alt="Runtime dependencies" src="https://img.shields.io/badge/runtime%20deps-none-555?style=flat-square&labelColor=0d1117">
<img alt="No GPU required" src="https://img.shields.io/badge/ISA%20build-no%20GPU%20required-555?style=flat-square&labelColor=0d1117">

<br/><br/>

<strong><a href="#the-problem">The problem</a> &nbsp;&middot;&nbsp; <a href="#how-it-works">How it works</a> &nbsp;&middot;&nbsp; <a href="#quickstart">Quickstart</a> &nbsp;&middot;&nbsp; <a href="#what-is-measured-not-assumed">What is measured</a> &nbsp;&middot;&nbsp; <a href="docs/ROADMAP.md">Roadmap</a> &nbsp;&middot;&nbsp; <a href="#clean-room-position">Clean-room</a></strong>

</div>

---

## The problem

An NVIDIA GPU instruction is 128 bits, and about 23 of them are not the instruction at all. They are a scheduling control word: how many cycles to stall before issuing the next instruction, which scoreboards to signal, which to wait on, and which operands may be served from the reuse cache.

**The hardware does not check any of it.** On sm_120 there is no interlock on fixed-latency instructions. The silicon trusts whatever produced the control word. If a stall count is shorter than the latency of a value the next instruction consumes, nothing faults, nothing stalls, and no warning is emitted. The instruction reads a register that has not been written yet and computes on stale data, at full speed, every single time.

That is a strange kind of bug. It does not crash. It does not appear in a debugger. It produces numbers that are merely wrong, which in a matrix multiply or an attention kernel means a model that trains slightly badly rather than one that visibly breaks.

Tools that generate machine code for this architecture *assign* those control bits from a latency model. basalt is the thing that checks the answer.

## How it works

Everything rests on two oracles, both of which are stock NVIDIA binaries driven as external processes. No NVIDIA source, headers, or libraries are used or redistributed.

| Oracle | Invocation | What it gives |
| :--- | :--- | :--- |
| **Ground truth** | `ptxas` → cubin → `nvdisasm -c -hex` | Encodings the vendor compiler actually emits. Semantics beyond dispute. |
| **Probe** | `nvdisasm -b SM120a` over raw bytes | Decodes words `ptxas` will never emit, which turns the encoding space into something searchable rather than something to guess at. |

The probe oracle is the one that matters. A tool limited to compiler output can only ever rediscover what the compiler already does. Feeding synthesised 128-bit words straight to the decoder means the instruction set can be *measured*.

Neither oracle needs a GPU, so the entire instruction database rebuilds in CI on any machine.

### Deriving the encoding by changing it

basalt does not read a table of opcodes from anywhere. It takes an encoding that assembled, flips one bit, decodes the result, and records what moved. A bit that changes the destination register is a destination bit; a bit that changes the mnemonic is a selector; a bit that changes nothing observable is inert.

Run against `IADD R5, R5, 0x2a`, the measurement comes out as:

```
operand[0]  bits 16:23     flip 16 -> R4,  flip 17 -> R7      destination register
operand[1]  bits 24:31     plus bit 72, which negates it      source register
operand[2]  bits 32:63     flip 32 -> 0x2b, flip 33 -> 0x28   32-bit immediate
opcode      bits 2, 4, 12:15
inert       36 bits        no observable effect
invalid     11 bits        the decoder rejects the mutation
```

Eight-bit register fields and a 32-bit immediate, arrived at by experiment rather than assumption.

### The control word

| Field | Bits | Meaning |
| :--- | :--- | :--- |
| `stall` | 108:105 | Cycles to wait before issuing the next instruction |
| `yield` | 109 | Hint that the warp scheduler may switch warps |
| `write_barrier` | 112:110 | Scoreboard to signal on write-back (7 = none) |
| `read_barrier` | 115:113 | Scoreboard to signal on operand read (7 = none) |
| `wait_mask` | 121:116 | Scoreboards that must be clear before issuing |
| `reuse` | 125:122 | Operand reuse-cache flags, one per source slot |

The layout validates itself on contact. In a trivial kernel, `S2R` sets `write_barrier=0` and the `IMAD` consuming its result carries `wait_mask=0x01`; `LDCU.64` sets `write_barrier=1` and the dependent `STG.E` carries `wait_mask=0x02`. Every producer and consumer pair lines up, and instructions that `nvdisasm` annotates `.reuse` have the matching reuse bit set.

## Quickstart

No CUDA installation and no GPU. The toolchain script fetches pinned redistributables, roughly 45 MB, no administrator rights, nothing added to your PATH.

```bash
git clone https://github.com/sunnypatell/basalt.git
cd basalt
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python scripts/fetch_toolchain.py     # pinned ptxas + nvdisasm
python -m basalt.cli doctor           # verify both oracles end to end
```

```console
$ python -m basalt.cli doctor
ok    toolchain   V13.3.73 in third_party/cuda/13.3.1/bin
ok    ptxas       assembled sm_120a
ok    cubin oracle  16 instructions with encodings
ok    probe oracle 16/16 mnemonics round-tripped

both oracles healthy. no GPU required for anything above.
```

Then rebuild the instruction database from scratch, or query the committed one:

```bash
python -m basalt.cli build-isa          # harvest, probe, write data/isa/sm_120a.json
python -m basalt.cli isa --stats
python -m basalt.cli isa IMAD.WIDE.U32  # one form, with its measured field layout
python -m basalt.cli isa --opcode QMMA  # every form of one opcode
```

## What is measured, not assumed

Numbers here are printed by the tooling and regenerate from a clean checkout. `basalt isa --stats` is the source of truth; this table is a snapshot.

| | |
| :--- | ---: |
| Instruction forms | 281 |
| Distinct opcodes | 84 |
| Forms with a full operand map | 276 |
| Tensor-core forms | 43 |
| Built with | `ptxas` V13.3.73 |

The tensor coverage is where the low-precision hardware lives: `HMMA` and `IMMA`, `QMMA` across the FP8, FP6 and FP4 types including asymmetric operand pairs, the scale-factor forms `QMMA.SF` and `OMMA.SF` that carry a per-block exponent, sparse `IMMA.SP`, and the matrix movement instructions `LDSM`, `STSM` and `MOVM` in every shape including the transposing variants.

> [!NOTE]
> **Alpha, and specific about what that means.** The instruction database is generated and grounded: every entry carries an encoding that really assembled and the compiler build that produced it. The hazard model and verifier are under construction, and the latency model they check against is not yet measured on silicon. Where something is inferred rather than measured, the tooling says so rather than rounding it up to a fact. See the [roadmap](docs/ROADMAP.md).

## Repository layout

```
src/basalt/
  toolchain.py     Locating and driving ptxas / nvdisasm
  encoding.py      The 128-bit instruction word and its control fields
  disasm.py        Both oracles: cubin ground truth and raw-word probe
  harvest/         PTX corpus generation and encoding extraction
  probe/           Differential bit probing and field inference
  isa/             The generated instruction database and its builder
  verify/          Register def-use analysis, hazard model, latency checking
data/isa/          Generated database, tracked so consumers need no harvest
docs/              Roadmap, method notes, findings, artwork sources
scripts/           Toolchain fetch, asset rendering, database drift check
tests/             Unit tests, plus toolchain- and GPU-marked suites
```

## Clean-room position

basalt is an independent, clean-room work for interoperability. It contains no NVIDIA source code, headers, libraries, or documentation, and redistributes none. It observes the behaviour of publicly distributed executables and records it, which is the footing this kind of work has stood on for over a decade.

NVIDIA, CUDA, and Blackwell are trademarks of NVIDIA Corporation. This project is not affiliated with, endorsed by, or sponsored by NVIDIA.

Licensed under [Apache-2.0](LICENSE). Apache rather than something restrictive on purpose: a correctness tool nobody is allowed to build on is a correctness tool nobody runs, and the patent grant matters for work this close to hardware.

## Contributing

The highest-value contribution is an encoding basalt gets wrong. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [ISA gap template](.github/ISSUE_TEMPLATE/isa_gap.yml), which collects enough to reproduce without your machine.

## Author

**Sunny Patel** &middot; [sunnypatel.net](https://www.sunnypatel.net) &middot; [ORCID 0009-0005-3863-7642](https://orcid.org/0009-0005-3863-7642) &middot; [github.com/sunnypatell](https://github.com/sunnypatell)
