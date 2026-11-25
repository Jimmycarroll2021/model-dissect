# Feature Specification: Weight editing for coding, without training

**Feature Branch**: `005-weight-edit-coding`
**Created**: 2026-11-25
**Status**: Draft
**Input**: User description: "Pull off something like Experiment 8 (a frontier LLM edits a small model's weights to make it better at math, no gradient descent) but for coding, on one of the little models, driven by Spec Kit."

Reference experiment: https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md

## The question

This repository already holds weight maps of three small models (MiniMind: 91-tensor inventory, editable reconstruction, input trace; Pythia-14m: weight atlas and intervention evidence; GPT-2: exhaustive atlas, attention trace, head interventions). Using those maps as the starting knowledge, can one of these models be made measurably better at code by editing its weights directly, with no gradient descent and no training?

The point is to put the mapping work to use: an edit is chosen because the map says what a component does, and the result tests whether the map was good enough to act on.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Baseline and model choice (Priority: P1)

The experimenter measures how well each candidate small model writes code today, and why it fails, and picks the model to edit by a rule fixed in advance.

**Why this priority**: The reference experiment worked because the model could already do the task and was failing for a fixable reason (93% of answers ran to the token limit). A model with no coding ability has nothing to unlock. Without this step the rest may be aimed at the wrong model.

**Independent Test**: Run the baseline on the practice set for every candidate; the output is a table of pass rate and failure types per model, and one chosen model with the rule that chose it.

**Acceptance Scenarios**:

1. **Given** the candidate list and the practice problems, **When** the baseline runs twice, **Then** both runs give identical results (decoding is greedy only).
2. **Given** baseline outputs, **When** failures are sorted, **Then** every failed answer has exactly one failure type (no code produced, syntax error, runtime error, wrong answer, timeout, cut at token limit, repetition loop).
3. **Given** the table, **When** the selection rule is applied, **Then** exactly one model is chosen, or the result is "no candidate qualifies" and the feature stops there with that reported.

---

### User Story 2 - An edit harness that enforces the rules (Priority: P2)

The editor (a frontier LLM) works through a harness that gives it the weights, the practice problems and a fixed number of forward passes, and that refuses anything forbidden.

**Why this priority**: The claim "no training" is only as good as the enforcement. The harness is also what makes a run replayable.

**Independent Test**: Scripted checks with no editor involved: a no-op edit leaves outputs unchanged; each forbidden action is refused; the pass counter stops at the budget; a saved edit rebuilds bit for bit from the public weights.

**Acceptance Scenarios**:

1. **Given** an edit that changes nothing, **When** the model is run, **Then** outputs are identical to the unedited model.
2. **Given** an attempt to edit the embedding, output layer or final norm, to scale a whole tensor by a constant, or to use gradients, **When** it is submitted, **Then** it is refused and logged.
3. **Given** the pass budget is used up, **When** another pass is requested, **Then** it is refused and the editor is asked for its final choice.
4. **Given** a finished run, **When** its recorded edit is applied to the public weights, **Then** the rebuilt tensors match the evaluated ones by hash.

---

### User Story 3 - One edit run and its held-out result (Priority: P3)

The editor makes its edit within one pass budget; the edit is frozen; the held-out coding benchmarks are scored once for the edited and unedited model.

**Why this priority**: This is the result. It depends on the first two stories.

**Independent Test**: Given a frozen edit, the evaluation produces one table (edited against unedited, per benchmark, with intervals) and the same table on a rerun.

**Acceptance Scenarios**:

1. **Given** a frozen edit, **When** the held-out benchmarks are scored, **Then** the report states pass rate for edited and unedited on each, the paired difference with an interval, and the share of answers cut at the token limit.
2. **Given** the same edit, **When** a size-matched random edit to the same tensors is scored, **Then** its result is reported beside the real edit.
3. **Given** the result, **When** it is written up, **Then** it is reported whether it is a gain, no change or a loss, with the full turn-by-turn log of the editor.

---

### User Story 4 - More budgets, second model (Priority: P4)

Repeat the run at other pass budgets and, if the first model's gain came from fixing a degenerate failure such as looping, on a model that does not have that failure.

**Why this priority**: Shows whether the effect scales with information and whether it is more than removing one bug. Only worth doing if Story 3 shows a gain.

**Independent Test**: Each extra run produces its own row in the same table.

**Acceptance Scenarios**:

1. **Given** runs at three budgets, **When** results are tabled, **Then** each has its own frozen edit, log and held-out score.

### Edge Cases

- No candidate model meets the selection rule: stop and report the baseline as the result.
- The editor's edit makes the model worse on practice: it may revert; the final choice may be "no edit", which is reported as such.
- The editor recognises a held-out problem from its own training: it never sees test problems or scores, but it knows these public benchmarks exist; reported as a limit.
- Generated code tries to use the network, write outside its folder, or run forever: the sandbox blocks or kills it and the answer counts as failed.
- A forward pass fails for a harness reason: it does not count against the budget and is logged.
- The gain is entirely from shorter answers (fewer cut at the limit): reported as that, not as better coding.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The candidate list, practice set, held-out sets, sampling settings and selection rule MUST be committed before any baseline is run.
- **FR-002**: Candidates MUST be the models already mapped in this repository (MiniMind, Pythia-14m, GPT-2). A model that is not mapped here may be added only as a fallback, after the baseline shows none of the three qualifies, and only together with the mapping work for it.
- **FR-002a**: Because these models are small, the baseline MUST measure code ability at two levels: whole problems passed (function passes its tests) and code completion (share of held-out code tokens or lines predicted correctly). The level used for the result is the hardest one on which the chosen model meets the selection rule.
- **FR-002b**: Each edit MUST cite the map entry (tensor, head, component and the evidence file in this repository) that motivated it.
- **FR-003**: The selection rule MUST be: the candidate with the fewest parameters whose practice score, at the hardest level where any candidate qualifies, is between 5% and 60%.
- **FR-004**: Practice problems MUST come only from training or validation splits; held-out problems MUST come only from test splits and MUST never be shown to the editor or used to choose anything.
- **FR-005**: Generated code MUST run only in a sandbox with no network, a time limit and a memory limit.
- **FR-006**: The editor MUST act only through the harness: a persistent compute session over the weights (no running the model, no gradients, no optimisers, no iterative fitting to a loss), and a forward-pass tool with a fixed budget.
- **FR-007**: The harness MUST refuse edits to the embedding, output layer and final norm, and edits that only scale a whole tensor.
- **FR-008**: One forward pass MUST be either a generation over at most 512 practice problems (original, edited or both, paired) or a teacher-forced read of at most 1,024 texts.
- **FR-009**: Every turn, code run, pass request and refusal MUST be logged, and the editor MUST explain each step in plain English before its code.
- **FR-010**: The final edit MUST be stored as an exact difference from the public weights, with hashes, and MUST rebuild bit for bit.
- **FR-011**: Held-out evaluation MUST run once per frozen edit, for edited and unedited weights with identical settings (greedy decoding), and be repeated once for identical numbers.
- **FR-012**: The evaluation MUST include a size-matched random edit to the same tensors and one non-coding check for side effects.
- **FR-013**: Every run started MUST be reported, including failed and abandoned ones.
- **FR-014**: The first run MUST use one budget (25 passes); other budgets belong to Story 4.

### Key Entities

- **Candidate model**: public name, pinned revision, size, weights hash.
- **Problem set**: source, split, role (practice or held-out), count, hash.
- **Edit**: tensors changed, entries changed, hashes before and after, the editor's stated reason.
- **Run**: editor, budget, turns, passes used, log, final choice.
- **Evaluation**: edit, benchmark, sampling settings, per-problem outcomes, summary.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Baseline pass rates for every candidate are reproduced by a second run.
- **SC-002**: All scripted harness checks pass: no-op identical, 100% of forbidden actions refused, budget enforced, edit rebuilds by hash.
- **SC-003**: The result counts as a gain only if the edited model beats the unedited model on at least one held-out coding benchmark by 3 percentage points or more with a 95% paired interval that excludes zero, beats the random edit by the same margin, and loses no more than 2 points on the side-effect check. Otherwise it is reported as no gain.
- **SC-004**: Someone with the public weights and this repository can rebuild the edited model and get the same held-out table.
- **SC-005**: The whole first run, from baseline to held-out table, fits on one 24 GB GPU.

## Assumptions

- The editor is Claude, run in its own session through the harness, with no access to this repository's held-out files.
- Practice set: MBPP train and validation. Held-out: HumanEval and MBPP test. Side-effect check: a small fixed sample of a general question benchmark. Exact versions are pinned in the plan.
- The mapped models are tiny and may pass no whole problems; that is why code completion is measured as a second level. GPT-2 is the most likely choice (largest of the three, English and some code in its training data).
- "Better at coding" means a higher share of problems whose generated function passes the benchmark's tests. Nothing is claimed about real software work.
- The reference experiment's gain came largely from removing a looping failure; a similar unlockable failure may or may not exist for coding. A null result is a valid outcome.
- The editor has probably seen the public benchmarks in its own training. This is a stated limit, not something this feature removes.
