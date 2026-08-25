# Ampere (`sm_86`) research boundary

Basalt does not have an Ampere control-word model. Its assembler, verifier,
scheduler, latency tables, and control-aware commands remain limited to the
measured `sm_120` family. Raw `nvdisasm` support for `SM86` is an oracle, not a
license to reuse the Blackwell field layout.

The repository does provide an observation-only migration path for studying a
new architecture without crossing that boundary.

## The three observation stages

1. `scripts/prototype_arch.py` retargets the architecture-neutral PTX corpus,
   compiles it, and records exact PTX, cubin, vendor disassembly, architecture,
   and runtime identities. It deliberately does not split a word into payload
   and control.
2. `scripts/mine_arch_fields.py` flips every one of the 128 bits in a
   deterministic representative set and records only visible raw-decoder
   effects. `scripts/verify_arch_fields.py` checks the complete bit domain,
   exact one-bit witnesses, and the report's claim ceiling.
3. `scripts/probe_arch_bits.py` patches exact cubin words and executes every
   original and mutant in an isolated process. Its campaign plan must bind the
   source cubins, target instruction text and bytes, device UUID/PCI identity,
   live lease, runtime objects, inputs, repetition count, timeout, and expected
   relation. The receipt keeps raw decode, module load, function lookup,
   launch, synchronization, output bytes, and timings separate.

These scripts are research surfaces. Their output is not an ISA database,
control schema, latency profile, scheduler input, or hazard verdict.

## What the first Ampere campaign found

The 2026-08-25 CPU corpus contained 36,296 naturally emitted `sm_86` words.
A deterministic set of 128 representatives produced 16,384 exact one-bit
mutations and rejected a global transplant of the `sm_120` layout. Upper-word
bits were visibly opcode-local across branches, matrix instructions, atomics,
and ordinary ALU instructions.

Bit 116 was the only bit that was both naturally variable and raw-decoder-text
silent for every representative. The GPU campaign falsified the stronger
hypothesis. These two exact words both decode as `MUFU.RCP R6, R6`:

```text
001e2400000010000000000600067308
000e2400000010000000000600067308
```

On an exactly identified RTX 3070, the original signed-division kernel returned
333 and -333 for the two fixed inputs. The bit-116 mutant returned 2 and -2.
Every value was stable across five repetitions. Bit 116 was output-equal in the
selected IADD3, predicated-loop-back-edge, and global-atomic witnesses, so the
division result was already a local counterexample rather than a global field
attribution.

A fresh follow-up then flipped bit 116 across direct O3 unary kernels for
`MUFU.COS`, `EX2`, `LG2`, `RCP`, `RSQ`, `SIN`, and `SQRT`, using two fixed
finite inputs per exact word. Every original and mutant loaded, launched,
synchronized, and produced identical deterministic bytes across five
repetitions. The direct `MUFU.RCP R0, R0` witness was output-equal even though
the earlier signed-division `MUFU.RCP R6, R6` witness diverged. All seven
mutations remained raw-decoder-text-silent, and the predicate detection control
again changed output.

The follow-up narrows the conclusion: the mnemonic alone is not the semantic
unit. Full encoding plus surrounding producer/consumer/control context is
load-bearing. This is not evidence that bit 116 is a dependency, yield, reuse,
stall, barrier, or scheduling field, and it does not establish semantic
equivalence outside the exact direct words and inputs.

A second fresh context matrix sampled eight previously unexecuted reciprocal
sites. Bit 116 changed deterministic output in optimized unsigned division,
signed remainder, and unsigned remainder, while the corresponding unoptimized
forms and two optimized floating-division sites remained output-equal. All
mutants loaded and synchronized; every pair was raw-decoder-text-silent. The
three divergent optimized integer forms used the same exact
`MUFU.RCP R6, R6` word and immediately consumed its result with `IADD3`. Their
unoptimized forms inserted a `MOV` before that consumer. But the direct
reciprocal witness had an immediate `FMUL` consumer and remained equal, so
distance alone is not established. Consumer class, full control encoding,
register arrangement, and dependency graph remain confounded.

Bit 122 also resisted global naming. Its one-bit mutation was raw-text-silent on
the selected atomic, added a `.reuse` source modifier to one BMMA, and made raw
`nvdisasm` reject another BMMA. Both selected BMMA mutants nevertheless loaded,
launched, synchronized, and returned the same deterministic output as their
originals. Decoder visibility, hardware legality, and selected-output equality
are therefore distinct observations. This result does not settle any external
bit-122 control-layout interpretation.

The campaign's predicate-bit detection control changed an unguarded IADD3 to
`@P6 IADD3` and changed the deterministic output, proving the runner was capable
of observing a semantic difference in the same cubin and execution path.

## Consequence

No `sm_86` field is promoted. In particular:

- raw decoder-text silence is not semantic silence;
- mnemonic equality is not context equality;
- optimization level and consumer neighborhood are evidence strata, not field names;
- a no-fault or output-equal mutation is not field attribution;
- hardware acceptance of a raw-decoder-rejected word is not general legality;
- one opcode's modifier spelling is not a cross-opcode field boundary;
- an Ampere execution observation cannot validate the Blackwell hazard model.

Control-aware `sm_86` operations must continue to fail before opening the
target. A future architecture profile requires a fresh, versioned campaign with
independent field witnesses, falsification probes, measured latencies, hazard
controls, cross-family refusal tests, and exact runtime/device authority.
