<div align="center">

<img src="./docs/assets/social-preview.png" alt="basalt: an assembler for NVIDIA consumer Blackwell sm_120 GPUs, turning SASS text into executable cubins" width="860" />

<h1>basalt &middot; a SASS assembler for sm_120</h1>

<p><strong>Write NVIDIA Blackwell machine code by hand. There has never been a public tool that lets you do this.</strong><br/>
Clean-room reverse engineering of the consumer Blackwell instruction set, an assembler that emits it, and a scheduler that places the control bits so the silicon does not silently corrupt your results.</p>

<img alt="Architecture" src="https://img.shields.io/badge/arch-sm__120%20%7C%20sm__120a-76B900?style=flat-square&labelColor=0d1117">
<img alt="Status" src="https://img.shields.io/badge/status-alpha-orange?style=flat-square&labelColor=0d1117">
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square&labelColor=0d1117"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=0d1117">
<img alt="No GPU required to build the database" src="https://img.shields.io/badge/ISA%20build-no%20GPU%20required-555?style=flat-square&labelColor=0d1117">

<br/><br/>

<strong><a href="#why-this-does-not-already-exist">Why it doesn't exist</a> &nbsp;&middot;&nbsp; <a href="#how-it-works">How it works</a> &nbsp;&middot;&nbsp; <a href="#quickstart">Quickstart</a> &nbsp;&middot;&nbsp; <a href="#status">Status</a> &nbsp;&middot;&nbsp; <a href="#legal-and-clean-room-position">Legal</a></strong>

</div>

---

> [!NOTE]
> **Alpha, and honest about it.** The ISA database is generated, not hand-written, and every claim below is reproducible from a pinned CUDA redistributable. Coverage numbers are printed by the tooling rather than asserted in prose. Where something is inferred rather than measured, it says so.

---

## What this is

NVIDIA GPUs execute **SASS**. NVIDIA does not document it, ships no assembler for it, and no public tool can produce it for consumer Blackwell. The supported path stops at PTX, a virtual ISA handed to `ptxas`, a closed-source compiler that decides what the hardware actually runs.

basalt goes the rest of the way. It reverse-engineers the sm_120 encoding from the vendor tools themselves, assembles SASS text into a real cubin, and solves the part that makes hand-written SASS lethal: **the control bits**.

On sm_120 there is no hardware interlock on fixed-latency instructions. If an instruction's stall count is shorter than the latency of the value it consumes, the hardware does not stall and does not fault. It reads a stale register and produces a wrong answer, quietly, at full speed. Every previous assembler in this lineage made a human place those bits by hand. basalt places them with a scheduler and then proves the placement safe.

## Why this does not already exist

Roughly once per architecture generation, one or two people reverse-engineer enough of SASS to write an assembler, and it is a small list: [maxas](https://github.com/NervanaSystems/maxas) for Maxwell, [turingas](https://github.com/daadaada/turingas) for Volta and Turing, [CuAssembler](https://github.com/cloudcores/CuAssembler) for several generations after. Scott Gray wrote maxas, used it to beat cuBLAS, and the work is still cited a decade later.

Consumer Blackwell has no such tool. The public state of the art is byte-patching instructions inside an existing cubin. Creating new SASS from scratch is not supported by anything public, and the people building adjacent tooling say so plainly.

That is the gap basalt fills.

## How it works

Everything rests on two oracles, both of which are stock NVIDIA binaries driven as subprocesses. No NVIDIA source, headers, or libraries are used or redistributed.

| Oracle | Invocation | What it gives us |
| :--- | :--- | :--- |
| **Ground truth** | `ptxas` → cubin → `nvdisasm -c -hex` | Encodings the vendor compiler actually emits. Semantics beyond dispute. |
| **Probe** | `nvdisasm -b SM120a` on raw bytes | Decodes encodings `ptxas` will never emit. This is how basalt reaches instructions and operand forms with no PTX spelling. |

The probe oracle is the one that matters. A harvester limited to compiler output can only ever rediscover what the compiler already does. Feeding synthesised 128-bit words straight to the decoder turns the ISA into something that can be *searched*, and the search finds capability the supported toolchain does not expose.

Both oracles run without a GPU, so the ISA database rebuilds in CI on any machine.

### The control bits

Each instruction carries a 23-bit scheduling section that the compiler normally owns. basalt re-derived the sm_120 layout and confirmed it against producer/consumer pairs in real compiler output:

| Field | Bits | Meaning |
| :--- | :--- | :--- |
| `stall` | 108:105 | Cycles to wait before issuing the next instruction |
| `yield` | 109 | Hint that the scheduler may switch warps |
| `write_barrier` | 112:110 | Scoreboard to signal on write-back (7 = none) |
| `read_barrier` | 115:113 | Scoreboard to signal on operand read (7 = none) |
| `wait_mask` | 121:116 | Scoreboards that must be clear before issuing |
| `reuse` | 125:122 | Operand reuse-cache flags, one per source slot |

The layout validates itself on contact. In a trivial kernel, `S2R` sets `write_barrier=0` and the `IMAD` consuming it carries `wait_mask=0x01`; `LDCU.64` sets `write_barrier=1` and the dependent `STG.E` carries `wait_mask=0x02`. Every producer/consumer pair lines up, and instructions nvdisasm annotates `.reuse` have the matching reuse bit set.

## Quickstart

No CUDA installation is required. The toolchain script fetches the pinned redistributables (about 45 MB, no admin rights, nothing added to PATH).

```bash
git clone https://github.com/sunnypatell/basalt.git
cd basalt
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python scripts/fetch_toolchain.py                   # pinned ptxas + nvdisasm
python -m basalt.cli doctor                         # verify both oracles
```

## Status

basalt is built in stages, each one independently useful. This table is generated from the test suite and updated as stages land.

| Stage | What it delivers | State |
| :--- | :--- | :--- |
| 1. Oracles | Round-trip harness over `ptxas` and `nvdisasm`, raw-word probe | **done** |
| 2. Harvest | Corpus generation and encoding extraction at scale | in progress |
| 3. ISA database | Validated opcode and operand-field tables for sm_120 | in progress |
| 4. Assembler | SASS text → cubin, patched back into a loadable module | planned |
| 5. Scheduler | Automatic control-bit placement | planned |
| 6. Verifier | Static proof that a program's stall counts cannot under-run latency | planned |

## Repository layout

```
src/basalt/
  toolchain.py     Locating and driving ptxas / nvdisasm
  encoding.py      The 128-bit instruction word and its control fields
  disasm.py        Both oracles: cubin ground truth and raw-word probe
  harvest/         Corpus generation and encoding extraction
  probe/           Systematic search of the encoding space
  isa/             The generated instruction database and its loader
  asm/             SASS text -> encoded word -> cubin
  sched/           Control-bit placement
  verify/          Static latency-safety checking
data/isa/          Generated ISA database (tracked; regenerable)
docs/              Method notes, control-bit experiments, findings
research/          Measurement logs and provenance
tests/             Unit tests, plus toolchain- and GPU-marked suites
```

## Legal and clean-room position

basalt is an independent, clean-room work for interoperability. It contains no NVIDIA source code, headers, libraries, or documentation, and redistributes none. It observes the behaviour of publicly distributed executables and records it, which is the same footing every prior assembler in this lineage has stood on for over a decade.

NVIDIA, CUDA, and Blackwell are trademarks of NVIDIA Corporation. This project is not affiliated with, endorsed by, or sponsored by NVIDIA.

Licensed under [Apache-2.0](LICENSE). Apache rather than a restrictive licence on purpose: a tool nobody is allowed to build on is a tool nobody uses, and the patent grant matters for work this close to hardware.

## Author

**Sunny Patel** &middot; [sunnypatel.net](https://www.sunnypatel.net) &middot; [ORCID 0009-0005-3863-7642](https://orcid.org/0009-0005-3863-7642) &middot; [github.com/sunnypatell](https://github.com/sunnypatell)
