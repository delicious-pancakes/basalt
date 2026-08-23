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

The version lives in exactly one place, and `CITATION.cff` carries it for
citation. Grep rather than trusting this list:

```bash
grep -rn "0\.1\.0" --include="*.toml" --include="*.cff" --include="*.md" .
```

`data/isa/sm_120a.json` carries its own `schema_version`, which tracks the
database format and is not the package version. Do not move it to match.

## 3. Changelog and commit

- Add `[X.Y.Z] - DATE` under `[Unreleased]` in `CHANGELOG.md`
  (Keep a Changelog, `### Added/Fixed/Changed`), add the compare link at the
  bottom, repoint `[Unreleased]`
- Commit: `chore(release): vX.Y.Z`, push `main`

## 4. Tag

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

The tag triggers `.github/workflows/release.yml`, which builds the sdist and
wheel, attaches SLSA build provenance through Sigstore, and uploads both to a
release. Nothing is published to an index from there, deliberately: an artifact
should be verifiable before it is distributed.

## 5. Verify the provenance before announcing

The whole argument of this project is that a claim without evidence is worth
nothing, and "these files came from that commit" is a claim like any other:

```bash
gh attestation verify basalt-X.Y.Z-py3-none-any.whl --repo sunnypatell/basalt
```

## 6. If a release is wrong

Yank rather than rewrite. Tags are cheap and a moved tag breaks every checkout
that already has it. Cut `X.Y.Z+1` with the fix and note in the changelog what
the previous release got wrong.
