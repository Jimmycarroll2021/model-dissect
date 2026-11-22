# Paraphrase v1 spec: does each audited edge hold on reworded prompts? (frozen before any intervention)

Frozen 2026-11-23. This file, `paraphrase-v1-corpora.json` and `paraphrase.py` are committed before the first intervention run. After freezing, no prompt is changed.

**Question.** Every own-prompt verdict in the pilot and scale audits rests on one prompt. This run asks whether each edge's measured effect holds on 5 paraphrases of that prompt.

**Population.** The 75 edges of the 25 testable graphs (5 in `graphs/`, 20 in `graphs2/`; `dallas-austin` and `gemma-it-pirate` are untestable and skipped). Edge selection is unchanged (`select_edges`, as in `audit2.py`). Own-prompt results are read from `results/pilot-v1-run1.json` and `results/scale-v1-run1.json`; they are not recomputed for the verdicts.

**Paraphrases.** `paraphrase-v1-corpora.json` holds 5 per graph: same language, same entity, same expected next token (the graph's top logit token, including its leading-space convention); only the framing is reworded. For `gemma-small-big-en` and `gemma-small-big-fr` the 5 are reused from the pilot variants that keep "small"/"petit" unchanged; all others are new.

**Position.** For every edge and paraphrase the feature is read, and its decoded contribution removed, at the FINAL token position of the paraphrase. The effect is measured on the final-position logits. Edges whose graph position is not the final token of the graph's own prompt (two: `gemma-bee-insect` 14_11504, `gemma-fact-shanghai-beijing` 19_12274 at the earlier token) are reported as POSITION_NOT_TRANSFERABLE and not run.

**Model and convention** as in `pilot-v1-spec.md` and `audit2.py`: google/gemma-2-2b, fp32, eager attention; features read from the pre-MLP RMSNorm output without its gain (input to pre_feedforward_layernorm divided by its RMS, eps 1e-6); the decoded contribution (coefficient times decoder row) is subtracted at the output of post_feedforward_layernorm; logits for E and the controls are taken at the lm_head output before the soft cap.

**Validity.** A paraphrase is valid for a graph if the target token's baseline probability (real, capped output) is >= 0.1. Invalid paraphrases are reported and not intervened on.

**Per valid paraphrase, per edge.** Coefficient of the feature at the final position. If active: E = drop in the target's mean-centred pre-cap logit when the contribution is removed; the 16 random-direction controls (PCG64 seeds 20261123 to 20261138, orthogonal to the decoder direction, scaled to the removed norm), their mean and max; M2 (unit decoder direction of the other active feature in the same layer and position with the nearest contribution norm, scaled to the removed norm); drop in the target's probability (baseline minus ablated, capped softmax).

**Per-paraphrase outcome** (pilot rule without the E/W band):
- INACTIVE: coefficient <= 1e-6.
- WRONG_SIGN: E <= 0.
- NOT_DISTINCT: E > 0 but any of E - C_R_mean >= 0.25, an M2 feature exists and E - M2 >= 0.25, E > C_R_max fails.
- DISTINCT: all three hold.
The edge's own-prompt E is reported as reference only and plays no part in the outcome.

**Summary per edge.** Number of valid paraphrases; fraction active; fraction DISTINCT; median E and median probability drop (inactive counts as 0); outcome counts. Paraphrase majority = "distinct" if DISTINCT count > half the valid count, otherwise "not distinct" (ties count as not distinct and are flagged). Own-prompt category = "distinct" if the own verdict is HOLDS, WEAKER or STRONGER, otherwise "not distinct".

**Overall.** Agreement = share of tested edges whose own-prompt category equals the paraphrase majority; the list of edges that disagree; per-paraphrase outcome counts over all valid edge-paraphrase pairs.

**Reproduction gate.** Before any paraphrase intervention the runner recomputes E for all 75 edges on the graphs' own prompts at the graph position and requires |E - stored E| <= 1e-3 for every edge; otherwise it stops and writes only the gate result.

**Repeat.** Run twice on GPU; result files must be identical (cmp).

**Baseline-only checks before freezing (no interventions).** First draft: `gemma-dollar` had 2 valid paraphrases and `gemma-english` 1. Their invalid paraphrases were rewritten (one-line framing headers such as "Country:currency" / "Country:language"; candidates screened on baseline probability only). `gemma-english` keeps one invalid paraphrase ("Analogy: ...") because the only other valid candidate found was a near-duplicate of an existing one. Final valid counts: 111 of 125; every graph has 3 to 5 (`gemma-euro`, `gemma-gp-nps`, `gemma-tom-brady` have 3).

**Not claimed.** Five paraphrases per graph is a small sample; the paraphrases are hand-written and share templates across graphs. Agreement measures whether the distinct / not-distinct call is stable under rewording, not whether the graph's weight W is right.
