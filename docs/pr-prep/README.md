# PR Prep Artifacts

This folder is the branch-local PR prep bundle for the current Paper Chaser change set.

## What is here

- `pr-description-draft.md`: long-form PR narrative with motivations, architecture, risks, and reviewer guidance
- `pr-body.md`: shorter PR body ready to paste into GitHub
- `commit-list.txt`: branch commits relative to the chosen base branch
- `changed-files.txt`: substantive changed-file list relative to the chosen base branch
- `diff-stat.txt`: substantive diffstat relative to the chosen base branch
- `full.diff`: substantive full patch relative to the chosen base branch
- `docs-and-packaging.diff`: docs/workflow/packaging slice of the diff
- `public-contract-and-runtime.diff`: public-contract/server/runtime slice of the diff
- `behavior-and-tests.diff`: behavior/tests slice of the diff

## Snapshot rules

The diff artifacts in this folder are generated against the current working tree relative to the chosen base branch, but they intentionally exclude `docs/pr-prep/**` from the comparison.

That exclusion is deliberate: otherwise the prep artifacts would recursively count and diff themselves, which makes file counts and patch size misleading for the actual PR.

## Usage

- Start with `pr-body.md` if you need a concise GitHub PR description.
- Use `pr-description-draft.md` when you want the full architectural and review framing.
- Use the diff slices when you want to review one area of the branch without reading the entire patch.
