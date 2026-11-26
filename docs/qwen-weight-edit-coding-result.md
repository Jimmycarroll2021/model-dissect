# Result: hand-editing Qwen3.5-0.8B to pass more whole coding problems

Spec 006, 2026-11-26. Status: **GAIN on held-out MBPP by the preset rule; no gain on HumanEval; maths unchanged.** Run twice, identical. Not independently rerun.

## Question

Can a model's weights be changed by hand, with no training, so that it passes more whole coding problems? Spec 005 could only ask this at the next-token level, because none of the mapped models can write a working function. Qwen3.5-0.8B can, so it was brought in as the fallback model, with its own mapping done inside the run (`exploratory/weightedit/qwen/runs/qwen-p25/notes.md`).

Protocols were frozen before the runs they govern: `specs/006-qwen-weight-edit-coding/{baseline-protocol.md,edit-protocol.md}`.

## Held-out result

Three sets of weights answered every held-out problem: unedited, edited, and a random edit of the same size (same changes in value, shuffled positions).

| Set | n | Unedited | Edited | Random edit | Edited minus unedited (95% interval) |
|---|---|---|---|---|---|
| MBPP 11 to 510 (same type as practice) | 500 | 13.8% (69) | 25.4% (127) | 14.8% (74) | +11.6 points (+8.6 to +14.8) |
| HumanEval (different type) | 164 | 32.3% (53) | 31.7% (52) | 31.7% (52) | -0.6 points (-1.8 to 0.0) |
| Maths, GSM8K (side-effect check) | 300 | 61.3% | 61.7% | 59.3% | +0.3 points (-1.3 to +2.0) |

Gain rule (set beforehand): at least 3 points over unedited with an interval excluding zero, at least 3 points over the random edit, maths down by no more than 2 points. MBPP meets it (+11.6 over unedited, +10.6 over random). HumanEval does not.

On MBPP the edited model gained 63 problems and lost 5.

## What the edit is

Two hand-written MLP neurons: 6,144 weights in 6 tensors, out of 752 million. 20 of the 25 allowed passes were used.

1. **Block 14, neuron 418: use the name the test calls.** It fires from `def` to the end of the function name and adds twice the average change that an explicit instruction ("name the function as the test does") makes to the model's internal state at those positions.
2. **Block 23, neuron 2621: stop after the function.** It fires when the next token would begin the model's own tests after the function, and raises the closing code fence over `#` and `assert`.

The model has no bias terms, so each neuron's threshold is built from a steady direction in the residual stream. Both neurons were put in slots that were idle across prompts, maths and code.

## Why it works on MBPP and not on HumanEval

The edit fixes two answer-format faults, not reasoning.

- Unedited, the function name the test calls was missing from 147 of 500 MBPP answers; edited, 58.
- Unedited, 415 of 500 answers carried the model's own tests or prints after the function, and 171 replies ran into the 512-token limit; edited, 55 and 116.

Of the 63 problems gained: 23 had only the wrong name, 23 had the right name but ran into the token limit, 6 had both, and 11 had neither (the model's own trailing tests or prints made the program fail). So **about half of the gain (29 of 63) is of the shorter-reply kind** the protocol warned about: it would shrink or vanish with a longer reply limit. The naming half would not.

HumanEval gives the function signature in the prompt, so the name is already right, and only 9 of 164 replies hit the limit. There was nothing for this edit to fix there.

## Tried and dropped

- Scaling up the attention heads that copy the name from the test (19.1, 19.5, 23.5) and scaling one down (23.3): fixed 13 name tokens, broke 11.
- An always-on version of the naming neuron: fixed names, but passes fell from 17 to 10 of 150 and maths slipped.
- The gated naming neuron at higher strength: at 24 almost every name token was wrong; at 4 the test's arguments were copied into the signature. Strength 2 was kept.
- First choice of slots for the two neurons: they were picked as idle on code only and were not idle on maths; re-picked on all positions.

## Practice numbers (for comparison)

Baseline: 73 of 464 practice problems (15.7%). Final check on 174 practice problems kept back from all editing: 33 to 55 passed, 22 gained, none lost.

## Limits

- One model, one prompt format, greedy decoding, a 512-token reply limit.
- The gain is format, not ability: the same effect is available by telling the model in the prompt (27 to 37 of 200 on practice with an instruction) or by cutting and renaming the answer afterwards.
- No gain on the different problem type.
- The editor (a Claude session) has probably seen MBPP and HumanEval in training; held-out isolation was by directory and code path.
- A pass here is up to 200 generated answers, a different unit from the reference experiment's.
- Not independently rerun.

## Files

- Edit: `exploratory/weightedit/qwen/runs/qwen-p25/{edit.npz,edit-manifest.json,notes.md,log.jsonl,kernel/}`
- Held-out verdict: `exploratory/weightedit/qwen/results/heldout.json`
- Code: `exploratory/weightedit/qwen/` (`qevaluate.py`, `run_heldout.sh`)
