# Feature Specification: hand-edit Qwen3.5-0.8B to pass more whole coding problems

**Feature Branch**: `006-qwen-weight-edit-coding`
**Created**: 2026-11-26
**Status**: Draft
**Input**: Jim, 2026-11-26: "Try Qwen?" after spec 005 ended with a small token-level gain on Pythia-14m and none of the three mapped models able to write a working function.

## Why

Spec 005 could only be judged on next-token prediction, because no mapped model can solve a coding problem. Qwen3.5-0.8B (the model the reference Experiment 8 edited for maths) answers coding questions in working Python, so the question can be asked at the level that matters: **whole problems passed**.

Qwen3.5-0.8B is not one of the mapped models. Under the standing goal it comes in as the fallback, with its own mapping work: here that is a tensor inventory and the activation readings taken inside the pass budget. Every edit cites the reading that motivated it.

## User Scenarios & Testing

### User Story 1 - Baseline (Priority: P1)

Measure how many practice problems the unedited model passes, how it fails, and how it does on a non-code check, under a protocol frozen first.

**Independent Test**: `run_baseline.sh` gives two identical result files and a go / no-go by the preset rule.

**Acceptance Scenarios**:
1. **Given** the frozen baseline protocol, **When** the baseline runs twice, **Then** the two result files are byte-identical.
2. **Given** the result, **When** the practice pass rate is between 5% and 80%, **Then** the edit run goes ahead; otherwise stop and report.

### User Story 2 - Edit run and held-out result (Priority: P2)

The editor (this Claude session) reads the model within a fixed pass budget, changes weights by hand with no training, freezes the edit, and the held-out problems are scored once.

**Independent Test**: `evaluate.py` on the frozen edit, run twice, identical, with a verdict from the preset rule.

**Acceptance Scenarios**:
1. **Given** a frozen edit, **When** held-out is scored, **Then** unedited, edited and a size-matched random edit are all reported with paired intervals.
2. **Given** any outcome, **Then** it is written up, including edits that were tried and dropped.

## Requirements

- **FR-001**: Model is `Qwen/Qwen3.5-0.8B`, text model only, weights hash recorded, fp32, greedy decoding, thinking off.
- **FR-002**: Success is measured on whole problems: generated code run against the problem's tests in the sandbox from spec 005.
- **FR-003**: Practice and held-out splits are those of spec 005. Held-out problems are not read or scored before the edit is frozen.
- **FR-004**: No gradients, optimisers or fitting to per-example targets; no edits to the embedding / output matrix (tied) or the final norm; no whole-tensor scaling.
- **FR-005**: Each edit records its reason and the measurement that motivated it.
- **FR-006**: The edit is exported as an exact difference from the published weights before evaluation.
- **FR-007**: Protocols are frozen (committed) before the runs they govern; every run that produces a reported number is repeated and must be identical.

## Success Criteria

- **SC-001**: Baseline result with failure types, reproduced exactly.
- **SC-002**: A gain is claimed only if, on held-out MBPP or HumanEval, the edited model passes at least 3.0 points more problems than the unedited one with a paired 95% interval that excludes zero, beats the size-matched random edit by at least 3.0 points, and the held-out maths solve rate falls by no more than 2.0 points. Otherwise the result is "no gain".
- **SC-003**: The write-up states what was tried, what was kept, and the limits.

## Assumptions

- The editor has probably seen MBPP and HumanEval in its own training; isolation of held-out is by directory and code path, as in spec 005.
- One model, one budget (25 passes). Other budgets only if this one shows a gain.
