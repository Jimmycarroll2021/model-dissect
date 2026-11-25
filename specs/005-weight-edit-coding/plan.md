# Implementation Plan: Weight editing for coding, without training

**Branch**: `005-weight-edit-coding` | **Date**: 2026-11-25 | **Spec**: [spec.md](spec.md)

## Summary
Measure the three mapped models on code at two levels, choose one by rule, edit its weights by hand using the repository's maps under a 25-pass budget, freeze the edit, score held-out once against the unedited model and a size-matched random edit.

## Technical Context
- Python 3 in the repository `.venv`, PyTorch, transformers; no new dependencies (datasets fetched as raw files with checksums).
- Code under `exploratory/weightedit/`; data, edits and generations under `/mnt/ai/model-dissect-data/weightedit/` (not in git); result summaries in `exploratory/weightedit/results/`.
- fp32, single GPU, greedy decoding, so runs repeat exactly.
- Sandbox for generated code: bubblewrap with all namespaces unshared (no network), read-only root, time and memory limits.

## Constitution Check
Freeze before measuring: `baseline-protocol.md`, then `edit-protocol.md` before the edit run, each committed first. Held-out: only `evaluate.py` reads held-out problems and only after an edit is frozen. Controls: unedited model and a size-matched random edit. Reproduce: every run twice. Smallest thing: plain scripts, no framework.

## Structure
```
exploratory/weightedit/
  common.py      model loading, tokenisation, greedy generation, scoring, sandbox
  fetch_data.py  download MBPP and HumanEval, split, write checksums
  baseline.py    Story 1
  harness.py     Story 2: pass budget, edit refusals, edit diff store, logs
  evaluate.py    Story 3: held-out scoring of unedited, edited, random edit
  results/       summaries committed to git
```

## Deviation from the spec, stated
The editor is the Claude session driving this repository, not a separate sandboxed session. Isolation of held-out data is by procedure and code path (the editor's tools cannot read held-out files; only `evaluate.py` can), and every pass and edit is logged by the harness. Reported as a limit in the write-up.

## Phases
1. Baseline and model choice (Story 1).
2. Harness with scripted checks (Story 2).
3. Edit run at 25 passes, freeze, held-out evaluation, write-up (Story 3).
4. Only if Story 3 shows a gain: other budgets (Story 4).
