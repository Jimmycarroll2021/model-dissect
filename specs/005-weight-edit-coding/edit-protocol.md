# Edit protocol (frozen before the first forward pass of the edit run)

Frozen 2026-11-25 with `exploratory/weightedit/{harness.py,harness_checks.py,evaluate.py}`.

**Model and level.** Pythia-14m, code completion (teacher-forced top-1 accuracy on code tokens), chosen by the baseline rule.

**Editor.** The Claude session driving this repository. It may read the repository's maps of this model (`docs/transfer-002-computation-map.md`, `docs/pythia-transfer-result.md`, `evidence/transfer-002/`), the weights, the practice problems and the non-code practice texts. It may not read anything under the held-out directory.

**Budget.** 25 forward passes. One pass is either
- a *score pass*: teacher-forced accuracy of the original and/or edited weights on at most 512 texts (practice code problems, non-code practice texts, or texts the editor writes), or
- a *read pass*: one set of weights run on at most 1,024 such texts, saving per-token inputs to every block, the per-head attention values before the output projection, MLP activations, the final residual, the predicted token and the log-probability of the true token.
A pass that fails for a harness reason does not count.

**Allowed without spending a pass.** Arithmetic on weights and on activations saved by read passes: norms, products of weight matrices, means and differences of means, covariances, SVD, applying a single weight matrix (including the final norm and output layer) to saved activations.

**Forbidden.** Gradients, autograd, optimisers; fitting weights to per-example targets (least squares or otherwise); search loops that score candidate weights on data outside the pass budget; running any block of the model outside the forward-pass tool; edits to the input embedding, output layer or final norm; edits that only multiply a whole tensor by a constant. `save_edit` refuses the last two groups and tensors that require gradients.

**Every edit** is saved with a stated reason and the map entry that motivated it.

**Final choice.** After the last pass (or earlier) the editor freezes the edit set, possibly empty. It is exported as an exact difference from the published weights (entries changed, float32 values, hashes of original and edited tensors) and committed before evaluation.

**Evaluation, once, after the freeze** (`evaluate.py`; two runs must be identical).
- Held-out code: MBPP task_id 11 to 510 (`# {text}\n{code}`, scored on the code) and HumanEval (`prompt + canonical_solution`, scored on the solution).
- Held-out non-code: GSM8K train questions 1001 to 1300 (never used in editing).
- Three sets of weights: unedited, edited, and a size-matched random edit: for each edited tensor the same changes in value, placed in randomly permuted positions among the changed entries (seed 20261125).
- Per problem: top-1 accuracy over scored tokens. Reported: mean per set of weights, paired difference with a 95% bootstrap interval (10,000 resamples of problems, seed 20261125).

**Gain rule.** The result is a gain only if, on held-out MBPP or HumanEval, edited minus unedited is at least 3.0 points with an interval that excludes zero, edited minus random edit is at least 3.0 points, and held-out non-code accuracy falls by no more than 2.0 points. Otherwise: no gain.

**Limits known now.** The editor is not sandboxed away from the repository; held-out isolation is by directory and code path. The editor has probably seen MBPP and HumanEval in its own training. Token-level completion is a weak proxy for writing code.
