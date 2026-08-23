# Security Policy

## Supported versions

basalt is pre-1.0 and alpha. Only the latest commit on `main` is supported.

## Release integrity

Every file attached to a release is built from a tagged commit by [`release-build.yml`](.github/workflows/release-build.yml) and signed through [Sigstore](https://www.sigstore.dev/), so you can check where a wheel came from instead of trusting the release page:

```console
$ gh attestation verify basalt_sass-<version>-py3-none-any.whl \
    --repo sunnypatell/basalt \
    --signer-workflow sunnypatell/basalt/.github/workflows/release-build.yml
```

That checks a [SLSA v1](https://slsa.dev/spec/v1.0/provenance) provenance statement naming the workflow, the commit and the runner that produced the file. `--signer-workflow` is optional; without it you still verify the repository and source commit, with it you also pin which workflow was allowed to build.

The builder is a separate reusable workflow rather than a job inside `release.yml`, and that is the reason the provenance is [Build L3 rather than L2](https://docs.github.com/actions/security-guides/using-artifact-attestations-and-reusable-workflows-to-achieve-slsa-v1-build-level-3): the build instructions run under their own identity, isolated from whatever dispatched them. The same split is the least-privilege boundary. The job that compiles, tests and signs has `contents: read`; only the separate publish job can write to a release.

`SHA256SUMS` covers the same files for anyone who would rather use `sha256sum -c`.

`basalt-sbom.cdx.json` is a [CycloneDX](https://cyclonedx.org/) bill of materials generated from a throwaway environment containing nothing but the wheel. It lists exactly one component, which is the machine-readable form of the claim that basalt has no runtime dependencies beyond the Python standard library. It is signed as a subject of the same attestation, so it is bound to those exact bytes rather than merely attached beside them.

### Supply-chain practices in the build

- **Every third-party action is SHA-pinned** with a trailing `# vX.Y.Z` comment, so a re-pointed tag cannot swap an action's code underneath a build. Dependabot keeps the pins fresh on a monthly cadence with a 7-day cooldown, so a compromised upstream release is usually withdrawn before a bump is even opened here.
- **The workflows are linted as code.** [actionlint](https://github.com/rhysd/actionlint) checks them for correctness and [zizmor](https://docs.zizmor.sh/) for template injection, credential persistence, unpinned actions and cache poisoning, on every change to `.github/`, at the same severity gate in CI and in pre-commit. The repository is currently clean at zizmor's lowest severity.
- **Least-privilege permissions.** Every workflow declares `permissions` at the top and every job that needs more says so locally. `id-token: write` and `attestations: write` exist only where signing happens.
- **The toolchain fetcher is checksum-gated.** `scripts/fetch_toolchain.py` verifies each NVIDIA redistributable's SHA256 against the published manifest and refuses to extract on a mismatch, so a substituted archive aborts the run instead of silently becoming the oracle everything else is measured against.
- **Egress is recorded.** Release builds run [harden-runner](https://github.com/step-security/harden-runner) in audit mode, so what the build talked to is checkable after the fact. Audit rather than block is deliberate: a blocking allowlist is a maintenance burden that tends to get widened until it means nothing, and a runner-resident agent is a detection layer, not a boundary.
- **CodeQL** runs on every pull request and weekly, [dependency review](.github/workflows/dependency-review.yml) blocks a pull request that introduces a known-vulnerable dependency, and [OpenSSF Scorecard](https://scorecard.dev/viewer/?uri=github.com/sunnypatell/basalt) re-scores the repository's posture weekly into the security tab.

### What this does not have

- **Bit-for-bit reproducible builds.** The wheel is pure Python and the inputs are pinned, but this is not verified byte-for-byte across independent rebuilds and no such claim is made. The provenance attestation is the compensating control: it binds each artefact to its source commit and builder.
- **A vulnerability scanner on the dependency graph.** basalt has no runtime dependencies, so a scan of the shipped SBOM has nothing to find by construction. Running one anyway would produce a green badge that means nothing. The dev and CI surface is covered by dependency review and Dependabot instead.
- **Hash-pinned Python dependencies.** Every GitHub Action is SHA-pinned, and the tools that touch the bytes that ship (`build`, `twine`, `cyclonedx-bom`) are exact-version pinned in [`release-build.yml`](.github/workflows/release-build.yml), as are the lint gate's `ruff` and `mypy`. What is not here is `pip install --require-hashes` against a fully-locked transitive manifest, which is what OpenSSF Scorecard's Pinned-Dependencies check looks for. basalt has zero runtime dependencies, so nothing pip installs reaches a user; maintaining a transitive lockfile to close a gap that ends at the CI runner is not a trade worth making. The Scorecard alerts for it are dismissed with this reasoning rather than left open.
- **Code-signing certificates.** There is no signed installer, because there is no installer: distribution is a wheel from a release page, and the Sigstore attestation is the thing to check.

## Reporting a vulnerability

Report privately through [GitHub Security Advisories](https://github.com/sunnypatell/basalt/security/advisories/new), or by email to sunnypatel124555@gmail.com. Please do not open a public issue for a vulnerability.

Expect an acknowledgement within 72 hours and an assessment within seven days.

## What counts as a vulnerability here

basalt is a developer tool that generates machine code. The realistic threat model is narrower than a network service, but two categories matter:

**Untrusted input reaching the assembler.** basalt parses SASS text, ISA database JSON, and cubin files. If any of those can be crafted to achieve code execution on the host, path traversal outside the output directory, or a crash that is exploitable rather than merely a traceback, that is a vulnerability.

**Supply chain.** The toolchain fetcher downloads NVIDIA redistributables over the network and verifies them against the checksums in the published manifest. A flaw that allows a substituted archive to be accepted is a vulnerability.

## What does not count

> [!NOTE]
> **Generating incorrect machine code is a correctness bug, not a security vulnerability.** It is a serious bug, and on this architecture it can corrupt data silently rather than fault, so please do report it. But report it as a normal issue using the ISA gap template so it can be discussed in the open, where other people working on the same encodings can see it.

Also out of scope: crashes on deliberately malformed input that produce a plain Python traceback, and anything requiring an attacker to already have write access to your machine.

## Scope

This policy covers the code in this repository. It does not cover NVIDIA's tools, which basalt drives as external processes. Vulnerabilities in `ptxas` or `nvdisasm` should go to NVIDIA.
