# Governance

Single-maintainer project: [@sunnypatell](https://github.com/sunnypatell) owns direction, review, releases and security response. No committee, no voting - honest about the bus factor of 1.

What that means in practice:

- **Decisions**: made by the maintainer, in the open (issues/PRs), with reasons written down
- **Contributions**: welcome via PR ([CONTRIBUTING.md](CONTRIBUTING.md)); contributor work is credited by handle in the changelog and release notes
- **Releases**: cut by the maintainer following [RELEASING.md](RELEASING.md); every artifact carries Sigstore-backed provenance ([SECURITY.md](SECURITY.md))
- **Security**: private disclosure per [SECURITY.md](SECURITY.md)
- **Continuity**: Apache-2.0, and every number in the documentation has a command in `scripts/` that recomputes it. A fork inherits a working repository rather than a snapshot of one

## What a change has to clear

The bar here is not taste, it is evidence, and it applies to the maintainer's own commits.

- **A claim needs a command.** Anything asserted in the README, `docs/FINDINGS.md` or a commit message names the script that produces it. A number with no way to recompute it is removed rather than trusted
- **The vendor compiler is the control.** A change that makes basalt disagree with `ptxas` on `ptxas` output is a regression, whatever else it improves
- **Derived beats assumed.** Where a constant can be mined from the corpus or measured on silicon, the mined value wins and the assumption stays only as a documented fallback
- **A refusal beats a guess.** An instruction basalt cannot encode from measured fields is declined with the field named. Emitting a plausible word is the one failure mode this project cannot have

## Scope changes

Adding an architecture, or claiming support for one, requires measurement on that silicon. The method generalises and the measurements do not, and shipping a latency model that was inferred rather than measured would reproduce exactly the failure basalt exists to catch. Proposals to widen scope are welcome; proposals to widen the *claim* without the measurement are not.
