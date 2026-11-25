# Tasks: Weight editing for coding, without training

## Phase 1: Baseline (US1)
- [x] T001 Freeze `baseline-protocol.md`, `common.py`, `fetch_data.py`, `baseline.py` in one commit before running
- [x] T002 Fetch data, record checksums and split counts
- [x] T003 Smoke: 8 practice problems per model end to end (both levels, sandbox)
- [x] T004 Full baseline, run twice, results identical
- [x] T005 Apply the selection rule; write `results/baseline.md`; commit

## Phase 2: Harness (US2)
- [x] T006 Write `edit-protocol.md` (budget, pass definition, refusals, gain rule at the chosen level) and `harness.py`; commit before any pass
- [x] T007 Scripted checks: no-op identical, each forbidden edit refused, budget enforced, edit rebuilds by hash

## Phase 3: Edit run and result (US3)
- [x] T008 Edit run: every edit cites its map entry; log every pass; final choice frozen and committed
- [x] T009 Held-out evaluation once: unedited, edited, size-matched random edit, non-code check; run twice
- [x] T010 Write-up `docs/weight-edit-coding-result.md` with limits; commit

## Phase 4 (US4, only after a gain)
- [ ] T011 Repeat at 10 and 50 passes (not started: Story 3 ended as NO_GAIN by the preset rule)
