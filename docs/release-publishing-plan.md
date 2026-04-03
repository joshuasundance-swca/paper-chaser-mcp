# Release And Publishing Plan

This document is the durable release playbook for the current
`paper-chaser-mcp` publishing model. It reflects the repo state after GHCR,
MCP Registry, GitHub Release assets, and dormant PyPI publication were split
into separate workflows.

Treat version strings in this document as placeholders. Use the checked-in
metadata in `pyproject.toml` and `server.json` to determine the current version
and the next version you intend to cut.

## Goals

The release process should:

1. Keep versioned releases immutable once tagged.
2. Make GHCR the primary public distribution channel for the containerized MCP package.
3. Publish wheel and sdist artifacts to the GitHub Release page for Python users who cannot or should not rely on PyPI yet.
4. Keep MCP Registry publication explicit and reviewable instead of coupling it to GHCR success.
5. Leave PyPI dormant until account recovery and trusted-publisher setup are complete.
6. Preserve the guided-default public story while keeping the expert surface available intentionally.

## Current Publishing Surfaces

| Surface | Workflow | Trigger | Current posture |
| --- | --- | --- | --- |
| GHCR image | `.github/workflows/publish-public-mcp-package.yml` | `v*` tag or manual dispatch | Primary public container distribution |
| GitHub Release assets | `.github/workflows/publish-github-release.yml` | `v*` tag or manual dispatch | Builds wheel/sdist, verifies with `twine check`, uploads to a draft Release |
| MCP Registry listing | `.github/workflows/publish-mcp-registry.yml` | Manual dispatch only | Run after the GHCR image is confirmed |
| PyPI / TestPyPI | `.github/workflows/publish-pypi.yml` | PR build, manual dispatch, `v*` tag | Build path is active; publish path remains gated behind `ENABLE_PYPI_PUBLISHING == 'true'` |

## Versioning Policy

1. Use stable semver-like PEP 440 versions in `pyproject.toml` and `server.json`, for example `<next-version>` such as `0.2.2`.
2. Use `v*` git tags to trigger publish workflows, for example `v<next-version>`.
3. Do not move or overwrite release tags after publication.
4. If a tagged release needs correction, cut a new version rather than rewriting the previous tag.
5. Prerelease tags (`-rc.1`, `-beta.1`, etc.) are supported by the current release workflows, but they should remain an explicit choice rather than the default release path.
6. If you intentionally cut a prerelease, keep the same exact prerelease identifier in `pyproject.toml`, `server.json`, and the git tag, for example `0.2.2rc1` in Python metadata paired with `v0.2.2-rc.1` only after verifying the packaging and registry surfaces you intend to ship can represent that version consistently.
7. Treat prerelease publishing as a narrower promotion path: GHCR should not receive the `latest` tag for prerelease builds, and release notes should make the prerelease status explicit.

## Release Phases

### Phase 1: Release-Prep PR

Create a dedicated release-prep branch from the repo's default branch and keep
the diff small.

Expected contents:

1. Bump `pyproject.toml` `project.version`.
2. Bump `pyproject.toml` `tool.bumpver.current_version`.
3. Bump `server.json` `version`.
4. Bump `server.json` GHCR package identifier to the same version.
5. Verify the guided/expert packaging story is still accurate:
   `mcp-tools.core.json` should still describe the guided default package,
   `mcp-tools.full.json` should still describe the expert package, and
   `microsoft-plugin.sample.json` should still point operators at the intended profile/package combination.
6. Decide whether the maturity classifier should remain Alpha or move to Beta.

Suggested command:

```bash
bumpver update --patch --no-commit --no-tag-commit --no-push
```

Use `--minor` or `--major` when the release scope warrants it.

Run focused validation immediately, then the normal release-prep validation
stack.

### Phase 2: Merge The Release-Prep PR

Do not tag from an unmerged branch. Merge the version bump first so all release
workflows operate on the committed default-branch contract.

### Phase 3: Tag The Release

From the up-to-date default branch:

```bash
VERSION=<next-version>
DEFAULT_BRANCH=<default-branch>

git switch $DEFAULT_BRANCH
git pull --ff-only origin $DEFAULT_BRANCH
git tag v$VERSION
git push origin v$VERSION
```

That tag should trigger:

1. GHCR publication through `.github/workflows/publish-public-mcp-package.yml`.
2. GitHub Release asset publication through `.github/workflows/publish-github-release.yml`.

It should not automatically trigger MCP Registry publication.

### Phase 4: Inspect Before Broad Promotion

After the tag workflows finish, inspect the outputs before treating the release
as broadly public.

Review checklist:

1. GHCR image exists at `ghcr.io/joshuasundance-swca/paper-chaser-mcp:<next-version>`.
2. GHCR tags look correct: exact version, major/minor tags, and `latest` only for non-prerelease releases.
3. OCI labels and MCP metadata match `server.json`.
4. The draft GitHub Release contains the wheel, sdist, and `SHA256SUMS`.
5. The wheel and sdist names match the version contract.
6. Release notes and presentation are acceptable for external users.
7. The package visibility state on GHCR is what you intend.
8. The guided-default messaging in README, packaging metadata, and workflow docs still matches the shipped product surface.

### Phase 5: Promote Discoverability

If the release looks good:

1. Make the GHCR package public if it is still private.
2. Publish the draft GitHub Release.
3. Run `.github/workflows/publish-mcp-registry.yml` manually for the same tag.

MCP Registry publication should happen only after the GHCR image and GitHub
Release assets are confirmed.

### Phase 6: PyPI Later

PyPI remains a separate concern.

Do not enable it until all of the following are true:

1. TestPyPI and PyPI account access is recovered.
2. Trusted publishers are registered for the final repo/workflow/environment identity.
3. The repository variable `ENABLE_PYPI_PUBLISHING` is intentionally set to `true`.

Until then, source installs and GitHub Release assets are the Python-user path.

## Release-Day Validation Checklist

Before tagging, run at minimum:

```bash
python -m pip check
pre-commit run --all-files
python -m pytest --cov=paper_chaser_mcp --cov-report=term-missing --cov-fail-under=85
python -m mypy --config-file pyproject.toml
python -m ruff check .
python -m bandit -c pyproject.toml -r paper_chaser_mcp
python -m build
python -m pip_audit . --progress-spinner off
```

If the release touches Docker or deployment-facing files, also run:

```bash
python scripts/validate_psrule_azure.py
python scripts/validate_deployment.py --skip-docker
```

If the release touches `.github/workflows/test-paper-chaser.md`, also rerun:

```bash
gh aw compile test-paper-chaser --dir .github/workflows
```

and commit the updated `.github/workflows/test-paper-chaser.lock.yml`.

## Local Workflow-Lint Parity

Workflow-file release changes are only trustworthy when local validation has CI
parity.

Rules:

1. Keep `shellcheck` available on `PATH` before running `pre-commit`.
2. Use redirected logs plus explicit completion checks for long pre-commit runs on Windows.
3. Do not treat partial integrated-terminal output as proof that validation is complete.

Recommended Windows pattern:

```bash
pre-commit run --files <changed-files> > precommit-release.log 2>&1
```

Then wait for completion, read the log, and only then commit or push.

## Recommended Sequence For The Next Release

Assuming the workflow and documentation groundwork is already merged:

1. Create and merge a small release-prep PR for `<next-version>`.
2. Tag `v<next-version>` from the default branch.
3. Inspect GHCR and the draft GitHub Release.
4. Make GHCR public and publish the GitHub Release if the artifacts look correct.
5. Run manual MCP Registry publication.
6. Leave PyPI gated unless the prerequisites above are intentionally completed.

## If Something Goes Wrong

If a release artifact or image is wrong after the tag is pushed:

1. Do not move the tag.
2. Fix the problem on the default branch.
3. Bump to the next version.
4. Retag with the new version.

This keeps the published record immutable and avoids ambiguity across GHCR,
GitHub Release assets, and MCP Registry metadata.
