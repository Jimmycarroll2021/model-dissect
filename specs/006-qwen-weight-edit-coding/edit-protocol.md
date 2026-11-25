# Edit protocol (frozen before the first forward pass of the edit run)

Frozen 2026-11-26 with `exploratory/weightedit/qwen/{qharness.py,qharness_checks.py,qevaluate.py}`.

**Amended 2026-11-26, before the first forward pass of the edit run.** The read pass can also save the per-head values that go into each attention output projection (`attn_z`), as the spec 005 harness did. Nothing else changed. Before this amendment the editor had re-judged the baseline's saved replies in the sandbox (no model pass): `runs/qwen-p25/kernel/k01` to `k03`.

**Model and level.** Qwen3.5-0.8B (text model), whole problems, prompts and judging as in `baseline-protocol.md`.

**Editor.** The Claude session driving this repository. It may read the weights, the practice problems (MBPP task_id 511 to 974), the practice maths questions (GSM8K train 1 to 300) and the baseline's replies to them. It may not read anything under the held-out directory.

**Mapping.** This model has no map in the repository. The editor's map is the tensor inventory plus what its read passes show; it is written into the run notes as it goes, and every edit cites the reading behind it.

**Budget.** 25 forward passes. The baseline's replies on the practice problems count as pass 1 (the run starts with 24 left). One pass is either
- a *score pass*: the original and/or edited weights answer at most 200 prompts (practice problems, practice maths questions, or prompts the editor writes), judged as in the baseline; or
- a *read pass*: one set of weights run teacher-forced on at most 256 texts (a user message plus a reply text, cut to 1,024 tokens), saving chosen internals per token (block inputs, MLP inputs, MLP activations, attention and MLP outputs, final residual), the predicted token and the log-probability of the true token.
A pass that fails for a harness reason does not count.

**Held-back practice.** Practice problems with task_id above 800 (174 problems) are not used for reading or scoring during editing. They may be scored once, as a final check before the freeze (it costs a pass). The harness refuses them otherwise.

**Allowed without spending a pass.** Arithmetic on weights and on activations saved by read passes: norms, products of weight matrices, means and differences of means, covariances, SVD, applying a single weight matrix (including the final norm and output matrix) to saved activations.

**Forbidden.** Gradients, autograd, optimisers; fitting weights to per-example targets (least squares or otherwise); search loops that score candidate weights on data outside the pass budget; running any block of the model outside the forward-pass tool; edits to the embedding / output matrix (tied) or the final norm; edits that only multiply a whole tensor by a constant. `save_edit` refuses the last two groups and tensors that require gradients.

**Every edit** is saved with a stated reason and the measurement that motivated it.

**Final choice.** After the last pass (or earlier) the editor freezes the edit set, possibly empty. It is exported as an exact difference from the published weights (entries changed, float32 values, hashes of original and edited tensors); the manifest is committed before evaluation.

**Evaluation, once, after the freeze** (`qevaluate.py`; run twice, results must be identical).
- Held-out code: MBPP task_id 11 to 510 (prompt as in the baseline) and HumanEval. HumanEval user message: `Complete this Python function. Reply with the full function in one code block.` followed by the problem's prompt in a python code fence. Program run: the problem's prompt, the extracted answer, the problem's test, `check(entry_point)`.
- Held-out maths: GSM8K train questions 1001 to 1300, prompt and judging as in the baseline.
- Three sets of weights: unedited, edited, and a size-matched random edit: for each edited tensor the same changes in value, placed at randomly permuted positions among the changed entries (seed 20261126).
- Reported per set: share passed for each set of weights, replies cut at the token limit, failure types, paired difference with a 95% bootstrap interval (10,000 resamples of problems, seed 20261126).

**Gain rule.** The result is a gain only if, on held-out MBPP or HumanEval, edited minus unedited is at least 3.0 points with an interval that excludes zero, edited minus random edit is at least 3.0 points, and the held-out maths solve rate falls by no more than 2.0 points. Otherwise: no gain.

**Limits known now.** The editor is not sandboxed away from the repository; held-out isolation is by directory and code path. The editor has probably seen MBPP and HumanEval in its own training. A pass here is a set of up to 200 generated answers, a different unit from the reference experiment's. One model, one prompt format, greedy decoding, a 512-token reply limit: a change that only makes replies shorter can raise the pass rate under this limit, and the report says how much of any gain is of that kind.
