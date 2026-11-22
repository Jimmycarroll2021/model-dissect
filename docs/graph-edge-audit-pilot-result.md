# Graph-edge audit: pilot v1 result

Run 2026-11-23 under `exploratory/pilot-v1-spec.md`, frozen in commit 4e92dab before any intervention. Exploratory lane: one run by one operator, repeated twice with identical output files, not independently rerun.

## What was tested

Five featured public Gemma-2-2b attribution graphs from Neuronpedia (Dallas/Austin, Oakland/Sacramento, small/big in English and French, 3 + 5). In each, the three transcoder features with the largest direct edge into the top output token: 15 edges. For each, the feature's contribution was removed in the real model on the graph's own prompt and the drop in the target's mean-centred logit (E) was compared with the graph's stated weight (W) and with size-matched controls.

## Findings

- **The setup matches the graphs.** All 15 feature activations computed here equal the graphs' values. For the one layer-25 edge the hand-computed direct path equals W exactly (4.122).
- **Direction: 15 of 15 correct.** Removing the feature lowered the target every time (smallest drop 0.02).
- **Size: usually smaller than stated.** Median E / W is 0.48. Five edges landed within a factor of two of W and beat every control (HOLDS). Four beat the controls but came in under half of W (WEAKER). Six did not clearly beat the controls (NOT_DISTINCT).
- **Rank order: moderate.** Spearman correlation between W and E is 0.52 across the 15 edges (p about 0.05; Pearson 0.39). Small sample.
- **The largest stated edge in each of three graphs was not the largest measured one.** Dallas 20/15589: W 5.31, E 1.37. Addition 24/14176: W 6.31, E 1.37, about the same as a random direction of that size (1.25). Small/big 23/8683: W 2.29, E 1.09.
- **Why six were not distinct.** In four of the six the only failed condition was exceeding the largest of 16 random draws; high-norm contributions in layers 23 and 24 make random edits of that size move the output a lot. One (21/10062) had almost no effect at all (E 0.02 against W 1.49).
- **Generality.** On paraphrase or same-answer variants the feature was active on every valid prompt for 13 of 15 edges; for 19/9209 on 62% and 21/10062 on 75%.
- **Specificity.** Seven edges left contrast and unrelated prompts untouched. Eight were active on other prompts and shifted them (largest top-two gap change 1.71), which fits general-purpose late-layer features rather than answer-specific ones.

Full table: `exploratory/results/pilot-v1-table.md`. Raw: `exploratory/results/pilot-v1-run1.json`.

## What this does and does not show

W is a direct-path weight in the replacement model; E is the total effect in the real model, where later layers can compensate or amplify. A gap is a statement about how the two relate, not an error in the graph. The verdict uses one prompt per edge; 15 edges from 5 graphs is a pilot, not an estimate for graphs in general. Logits are taken before the model's final soft cap, the unit the graphs use; after the cap the drops are smaller.

## Earlier result, corrected

The first runs of this project reported an 8.4-logit drop for 20/15589. That figure came from editing the wrong point in the model and is withdrawn; see the correction in `docs/graph-edge-audit-goal.md`.
