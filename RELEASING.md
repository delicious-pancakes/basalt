# Releasing basalt

The runbook for cutting a release. The workflow does the building and the
signing; this file exists so a release never depends on memory.

## 0. Preconditions

- On `main`, clean tree
- `gh` authenticated as the repo owner
- Every control green. One command, and it says which steps it skipped:

```bash
python scripts/verify_all.py
```

  On a machine with no sm_120 card the hardware steps report as **skipped**,
  not passed. A release cut from such a machine is legitimate; a release that
  quietly counted a skipped control as a pass is not.

## 1. Regenerate what the documentation quotes

Every published figure has a command behind it, and a release is the moment
those had better still agree:

```bash
python scripts/corpus_figures.py       # the corpus and coverage numbers
python scripts/rebuild_and_compare.py  # the ISA database has not drifted
```

If a number moved, the documentation changes with it in the same commit. A
figure the repository can no longer reproduce is removed rather than kept.

## 2. Bump the version

The version lives in `pyproject.toml`, in `src/basalt/__init__.py` for
`--version`, and in `CITATION.cff` for citation. `tests/test_packaging.py` fails
if the first two disagree, so the grep is a cross-check rather than the guard:

```bash
grep -rn "X\.Y\.Z" --include="*.toml" --include="*.cff" --include="*.md" --include="*.py" .
```

`src/basalt/data/isa/sm_120a.json` carries its own `schema_version`, which tracks the
database format and is not the package version. Do not move it to match.

## 3. Changelog and commit

- Add `[X.Y.Z] - DATE` under `[Unreleased]` in `CHANGELOG.md`
  (Keep a Changelog, `### Added/Fixed/Changed`), add the compare link at the
  bottom, repoint `[Unreleased]`
- Commit: `chore(release): vX.Y.Z`, push `main`

## 4. Enable Zenodo, once, before the first tag

Zenodo mints the DOI from a GitHub release, so the integration has to be on
*before* the tag is pushed or that release gets no DOI. It is a one-time switch
and only the repository owner can flip it.

1. Sign in at [zenodo.org](https://zenodo.org) with GitHub
2. Under **GitHub**, find `sunnypatell/basalt` and turn the toggle on
3. Check `CITATION.cff` has the right `version` and `date-released`

Zenodo issues two DOIs: one for the specific version and a **concept DOI** that
always resolves to the newest. The concept DOI is the one to put in the README
badge and in `CITATION.cff`, because it never has to change again.

## 5. Write the release, then tag

The workflow attaches artefacts to a release; it does not write one. Create it
first so the body is yours rather than generated:

```bash
gh release create vX.Y.Z --draft --title "vX.Y.Z" --notes-file NOTES.md
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

Draft first is deliberate. The release stays invisible until the artefacts are
attached, and the tag is validated against a semver allowlist before a build
minute is spent, so a typo fails in five seconds rather than after the matrix.
Before anything is uploaded the pipeline checks that there is exactly one wheel
and one sdist, that `SHA256SUMS` and the SBOM are present, that neither
distribution is small enough to have lost its data, and that the wheel actually
contains the three measured tables. That last check exists because a wheel once
shipped without them: it passed every test and was useless, because installed
anywhere but a checkout it fell back to assumed latencies and then reported
hazards it could not ground.

The tag triggers `.github/workflows/release.yml`, which calls the reusable
`release-build.yml` to build the sdist and wheel, run the GPU-free suite,
generate the SBOM and sign all three through Sigstore, then publishes what came
back from a separate job. The split is what makes the provenance SLSA Build L3,
and it means the job that builds never holds `contents: write`. Nothing is
published to an index from there, deliberately: an artifact should be verifiable
before it is distributed.

## 6. Verify the provenance before announcing

The whole argument of this project is that a claim without evidence is worth
nothing, and "these files came from that commit" is a claim like any other:

```bash
gh attestation verify basalt_sass-X.Y.Z-py3-none-any.whl \
  --repo sunnypatell/basalt \
  --signer-workflow sunnypatell/basalt/.github/workflows/release-build.yml
```

## 7. After the DOI exists

Zenodo mints it within a minute of the release being published. Then, in one
commit:

- add the concept DOI badge to the README
- add `doi:` and an `identifiers:` block to `CITATION.cff`

A post-release commit for this is normal and expected; the DOI cannot exist
before the release it describes.

## 8. If a release is wrong

Yank rather than rewrite. Tags are cheap and a moved tag breaks every checkout
that already has it. Cut `X.Y.Z+1` with the fix and note in the changelog what
the previous release got wrong.
