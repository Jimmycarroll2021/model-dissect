# Graph-edge audit: result over every featured public Gemma-2-2b graph

Run 2026-11-23. Specs frozen in git before any intervention: `exploratory/pilot-v1-spec.md` (commit 4e92dab) and `exploratory/scale-v1-spec.md` (commit cef330c). Exploratory lane: one operator, each run repeated with identical output files, not yet independently rerun.

## Question

Public attribution graphs state, for an edge, "this feature pushes the model toward this answer with weight W". W is computed in a replacement model. Does removing that feature in the real Gemma-2-2b lower the answer, by how much, and by more than a same-size edit of no particular meaning?

## What was tested

All 27 graphs featured on Neuronpedia for gemma-2-2b with the gemmascope-transcoder-16k dictionary. In each, the three features with the largest direct edge into the top output token. Two graphs could not be tested (one older file records no activations and uses a different weight scale; one chat-format prompt does not tokenise to its recorded tokens). That leaves **25 graphs and 75 edges**.

Setup check: on all 75 edges the feature activation computed here equals the graph's recorded value; for final-layer edges the hand-computed direct path equals W.

E is the drop in the target's mean-centred, pre-soft-cap logit on the graph's own prompt when the feature's contribution is removed. Controls: 16 random directions of the same size, and another active feature's direction at the same size.

## Result

| | edges | share |
|---|---|---|
| HOLDS (beats every control, E within a factor of two of W) | 20 | 27% |
| STRONGER (beats controls, E more than twice W) | 1 | 1% |
| WEAKER (beats controls, E under half of W) | 15 | 20% |
| NOT_DISTINCT (right sign, does not clearly beat same-size controls) | 30 | 40% |
| WRONG_SIGN (answer did not fall) | 9 | 12% |

- **Direction:** right on 66 of 75 edges (88%).
- **Size:** median E / W is 0.38 (middle half 0.15 to 0.71). Measured effects are usually well under the stated weight.
- **Ranking:** Spearman correlation of W with E is 0.31 (p 0.006), Pearson 0.27. In 13 of 25 graphs the edge with the largest W was also the one with the largest E (chance is one in three).
- **By layer:** median E / W is 0.14 for features below layer 20 (13 edges, 4 with the wrong sign), 0.32 for layers 20 to 22 (25 edges), 0.48 for layers 23 to 25 (37 edges). The further a feature sits from the output, the less its stated weight says about the real effect.
- **Not distinct:** 9 of the 30 failed only the strictest condition (beating the largest of 16 random draws); the other 21 also failed against the random mean or the size-matched feature direction.
- **Wrong sign:** all three edges of the "A bee is a type of" graph, two of the "Mexico:peso :: US:" graph, and four others. Each has a positive hand-computed direct path, so the rest of the model more than offsets the removal.
- **Specificity:** of 73 final-position edges, 38 left every other prompt's top-two gap within 0.1; 35 features also fire on other prompts and move them.
- **Generality (pilot only, 15 edges):** the feature stayed active on every valid paraphrase for 13 edges.
- After the model's final soft cap the same drops are smaller (median about 0.4 of the pre-cap value).

Full per-edge table: `exploratory/results/audit-all-table.md`. Raw: `exploratory/results/pilot-v1-run1.json`, `exploratory/results/scale-v1-run1.json`.

## Follow-up on the nine wrong-sign edges (after the fact, exploratory)

`exploratory/wrongsign_probe.py`, results in `exploratory/results/wrongsign-probe.json`. Not preregistered.

- **They are near-zero effects, not reversals.** The centred-logit change is small in all nine (0.04 to 1.0), it scales smoothly with how much of the feature is removed, and the answer's probability still fell slightly in 8 of the 9 (by 0.002 to 0.033; it rose by 0.021 for bee 19/5460). "Wrong sign" is a property of the mean-centred logit measure here; a fairer label is "no real effect".
- **Later layers undo most of the push.** Reading the change in the residual stream layer by layer through the frozen final norm: for bee 22/16038 the removal is worth 4.74 logits just after layer 22 (the hand-computed direct path), 2.56 after layer 23, 1.60 after 24 and 1.46 after 25. Seven of the other eight also end lower than they start; the one earlier-token edge (bee 14/11504) builds to about 1.0 by layer 19 and falls to 0.38.
- **The final normalisation accounts for the rest.** In all nine the drop with the final norm frozen is still positive after layer 25 (0.07 to 1.46) while the measured change is a small rise, so the sign flips inside the final norm. The likely mechanism (inferred, not separately measured): removing a large contribution shrinks the residual, the norm rescales everything up, and the mean-centred logit of an already-high answer rises.
- **The same decline shows in edges that hold**, only less: Dallas 20/15589 goes from 3.52 after its own layer to 1.41 after layer 25 (measured E 1.37).

So the gap between stated weight and measured effect is mostly downstream layers compensating, plus a rescaling effect in how the final norm treats a removed vector. Both are real properties of the model that a direct-path weight does not include.

## Independent re-measurement (2026-11-23)

A separate reviewer wrote its own code from the two spec files and the graph files only, without access to the audit runners or results, and re-measured 15 edges chosen to include the surprising ones (files in `exploratory/independent-check/`).

- **The measured effects reproduce.** E agrees on all 15 edges to within 0.00002, before and after the soft cap. All 15 activations and baseline probabilities match the graphs. The reviewer confirmed by test that the read point, write point, mean-centring and pre-cap unit are the ones that reproduce the graphs' own numbers.
- **The random control is flawed.** The spec does not say whether the random vector is added or subtracted; the audit added it, the reviewer subtracted it, and the control values differ (for example Dallas 20/15589: mean 0.11 here, 0.37 there). More important, a random vector of that size makes the residual larger, which lowers every logit through the final norm, while removing a real feature makes it smaller. So the random control carries a norm effect the real ablation does not, and it is biased upward for large features. The split between WEAKER/HOLDS and NOT_DISTINCT therefore depends on a control that is not like-for-like; at that point the NOT_DISTINCT count was read as an upper bound; the like-for-like control run afterwards (below) shows that reading was wrong.
- **The strictest rule is noisy.** "Beat the largest of 16 random draws" on one prompt is the binding condition for most edges.
- **Which number is the headline matters.** The pre-cap E is the right unit for comparing with W, but after the soft cap the drops are 3 to 20 times smaller for confident answers, and no edge among the 15 moved its answer's probability by more than 0.30.

What survives unchanged: the measured E values, their sign, E / W, the correlation with W, and the layer pattern. What is weakened: the exact HOLDS / WEAKER / NOT_DISTINCT split.

## Paraphrase check (2026-11-23, frozen in `exploratory/paraphrase-v1-spec.md`)

Each edge was retested at the final position of up to 5 reworded prompts with the same expected answer (111 valid paraphrases over 25 graphs; 73 edges, 2 skipped because their feature sits at an earlier token). Two runs identical; the runner first reproduced all 75 own-prompt E values exactly.

- The feature was active on 319 of 324 edge-paraphrase pairs.
- E > 0 on 268 of 319 active pairs (84%).
- Per-edge median E over paraphrases tracks the own-prompt E: Spearman 0.94; same sign on 71 of 73 edges. Median of (median E) / W is 0.40, against 0.38 on the own prompts. Spearman of W with median E is 0.35.
- The distinct / not-distinct call (v1 controls) agrees between own prompt and paraphrase majority on 67 of 73 edges.
- Individual paraphrases do vary: one addition edge ranges from -6.0 to +2.9 across five framings.

Table: `exploratory/results/paraphrase-v1-table.md`.

## Like-for-like control (2026-11-24, frozen in `exploratory/control-v2-spec.md`)

Made after seeing the v1 results, to answer the reviewer's objection. The control vector has the same length as the removed contribution and the same projection onto the residual stream, with the rest in a random direction, and is subtracted, so the edited residual has exactly the same norm as under the real ablation. 64 draws per edge; an edge is distinct if E beats the control mean by 0.25 and at most 3 of 64 draws reach E. Two runs identical; all 75 E values reproduced.

- Verdicts: HOLDS 17, WEAKER 11, NOT_DISTINCT 38, WRONG_SIGN 9 (v1: 20 + 1 stronger, 15, 30, 9). Twelve edges change, ten of them towards NOT_DISTINCT.
- The matched control is harder to beat, not easier: its median effect is 0.32 logits against 0.05 for the v1 control. Removing the part of a contribution that lies along the residual stream lowers the leading answer on its own, and the removed contributions have a median 17% of their length along it.
- So the reviewer's expectation, and the "upper bound" reading above, were wrong in direction. Among the 66 right-sign edges, 53 beat the control mean; 28 pass the full rule.

The split into holds / weaker / not distinct depends on which same-size edit counts as fair: 36 of 75 edges are distinct under v1, 28 under v2. Sign, E / W, the correlation with W and the layer pattern do not depend on the control.

## Reading

For these featured graphs the direction of a top edge is usually right, but its weight is a poor guide to what removing the feature does in the real model: the measured effect is typically about 40% of the stated weight, and between 37% and 48% of top edges (depending on the control) are clearly distinguishable from a same-size edit; about one in eight goes the other way. The picture is the same on reworded prompts. Edges in the last three layers are the most reliable.

## Limits

- W is a direct-path weight; E is a total effect including downstream compensation. A gap is a fact about how the two relate, not an error in the graph. The graphs do not claim W predicts ablation.
- One prompt per edge for the verdict (paraphrases checked separately, above); 25 hand-picked featured graphs, top three edges only. Not a sample of graphs in general.
- Removing a feature to zero is one intervention; steering or patching could behave differently.
- The size-matched feature control pushes that feature past zero, which is a harsh comparison.
- The pass marks (0.25 logits, factor of two) are choices made before the runs, not derived quantities.
- One model, one dictionary, fp32 on one GPU. Independent re-measurement covered 15 of 75 edges.

## Withdrawn

An earlier figure of an 8.4-logit drop for feature 20/15589 came from editing the wrong point in the model and is withdrawn; see `docs/graph-edge-audit-goal.md`.
