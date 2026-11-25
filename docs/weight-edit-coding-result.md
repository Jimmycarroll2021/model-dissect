# Editing a mapped model's weights by hand to improve code prediction

Spec: `specs/005-weight-edit-coding/`. Run 2026-11-25. Exploratory lane; one run, repeated for identical numbers, not independently rerun.

## Question

This repository holds weight maps of MiniMind, Pythia-14m and GPT-2. Using those maps, can one of the models be made measurably better at code by editing weights directly, with no gradient descent and no training?

Modelled on Experiment 8 of `RikiyaT/lm_understand-logs` (a frontier model edits a small model's weights to improve maths), with the editor being the Claude session driving this repository.

## Short answer

Partly. Eight hand-written neurons (2,056 changed weights out of 14.1 million) raised Pythia-14m's code-token accuracy on held-out MBPP problems from 40.1% to 43.0%, a gain of **2.8 points** (95% interval 2.6 to 3.1), with no change on English text. That is a real, repeatable improvement, but it is below the 3-point mark fixed before the run, so by the preset rule the verdict is **NO_GAIN**. On HumanEval, which is laid out differently, nothing changed (-0.2, interval -0.5 to +0.1). The edit taught the model the layout and naming habits of one style of problem; it did not make it better at code in general.

## Baseline and model choice

464 MBPP practice problems, greedy decoding (`exploratory/weightedit/results/baseline.md`).

| Model | Weights | Whole problems passed | Code tokens predicted | English tokens predicted |
|---|---:|---:|---:|---:|
| MiniMind | 63.9M | 2 of 464 | 61.0% | 42.6% |
| Pythia-14m | 14.1M | 0 | 40.3% | 25.5% |
| GPT-2 | 124.4M | 0 | 47.5% | 37.4% |

No model can write working functions, so the result is judged on code completion: the share of code tokens predicted correctly when the true code is fed in. The rule fixed in advance (fewest weights among models scoring 5% to 60%) chose Pythia-14m.

## What the editor could do

Frozen in `specs/005-weight-edit-coding/edit-protocol.md` before the first pass: 25 forward passes (10 were used); free arithmetic on the weights and on saved activations; no gradients, optimisers, fitting to targets or offline scoring loops; no edits to the embedding, output layer or final norm. Held-out problems live in a directory only the evaluator reads. The harness refuses forbidden edits and enforces the budget (`harness_checks.py`, all pass).

## What was tried

Full turn-by-turn record: `exploratory/weightedit/runs/pythia-p25/notes.md`, with every script and output under `kernel/` and every pass under `requests/`.

1. **Pruning by attribution: failed.** The computation map allows each head's and each MLP neuron's write to be read out through the final norm and output layer. By that reading, 26 neurons in the last MLP lowered the right token on code, stably across two halves of the data. Switching them off made code accuracy 3.1 points *worse*. The attribution compared the right token with one rival; removing a neuron moves every token. Reverted.
2. **Hand-written neurons: worked.** For a context where one next token is nearly always right and the model misses it, take the difference of mean MLP input between that context and everything else as a detector, set a threshold so the neuron is silent elsewhere, and make its output the direction of the right token. The map supplies exactly what is needed: which row a neuron reads, which column it writes, and that the last block writes straight to the output.

The final edit, with what each neuron does:

| Layer | Fires when | Raises | Kind of knowledge |
|---|---|---|---|
| 5 | end of the opening `# description` line | `def` | layout of these problems |
| 5 | end of a top-of-file import line | `def` | layout |
| 3 | one token after `def` | `_` | snake_case names |
| 3 | after `for i in` | ` range` | Python idiom |
| 3 | after `(test` | `_` | this set's argument names |
| 5 | after `res)` | space | this set's style |
| 5 | after `(res` | `)` | this set's style |
| 5 | top-of-file `import` | ` re` | this set's habits |

The first two rules account for most of the gain. Three things did not work and were removed: a `return` rule (ambiguous context), a rule placed in layer 4 (its write was damped by layer 5 until moved there), and a last round of three small rules (net negative).

## Held-out result

Scored once after the edit was frozen and committed; two runs identical (`exploratory/weightedit/results/heldout.json`).

| Set | Problems | Unedited | Edited | Random edit | Edited minus unedited (95% interval) |
|---|---:|---:|---:|---:|---|
| MBPP test | 500 | 40.11% | 42.95% | 39.99% | +2.83 (2.57 to 3.10) |
| HumanEval | 164 | 47.39% | 47.22% | 47.10% | -0.18 (-0.51 to +0.13) |
| English (GSM8K questions) | 300 | 25.81% | 25.84% | 25.84% | +0.03 (-0.17 to +0.23) |

The random edit puts the same changes in shuffled positions within the same tensors. It behaves like switching the eight neurons off: slightly worse than unedited on code.

Against the preset rule (at least +3.0 on a held-out code set, +3.0 over the random edit, English down no more than 2.0): MBPP reaches +2.83 and +2.96. **NO_GAIN by the rule; a clear gain short of the mark.**

## Did the maps hold up?

- The static computation map was accurate enough to act on: every installed neuron read and wrote where the map said, and last-layer writes had the predicted direct effect.
- The map says nothing about what downstream layers do with a write. The layer-4 rule failed for that reason, and the layer-3 rule needed twice the computed strength.
- Reading components through the output layer (the natural use of the map for diagnosis) pointed the wrong way for pruning.

## Limits

- Token-level completion is a weak proxy for writing code. No model here writes working functions before or after.
- The gain is specific to the MBPP layout and its authors' habits; HumanEval shows no transfer.
- The detectors and rules were derived from 300 practice problems by counting and mean differences. No gradients were used, but this is still learning from data, done by hand.
- The editor was not sandboxed from the repository; held-out isolation was by directory and code path. The editor has very likely seen MBPP and HumanEval in its own training.
- A code review during setup found the sandbox for generated code had never started; it was fixed before any result was recorded.
- One model, one run, greedy decoding, one GPU. Not independently rerun.

## Follow-up worth doing

MiniMind is the only one of the three that writes recognisable functions, and 58% of its answers fail by looping, the same failure the reference experiment removed. The preset rule excluded it here by one point of token accuracy. It is the natural candidate for a whole-problem version of this experiment.
