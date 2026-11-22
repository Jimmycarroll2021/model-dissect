# Held-out confirmation spec v2 (frozen before any forward pass on this corpus)

Frozen 2026-11-23. Commit of this file and `corpus-v2-heldout.json` precedes the first run.

**Claim.** In Gemma-2-2b (fp32, eager attention), ablating GemmaScope layer-20 transcoder feature 15589's decoded contribution at the final input token lowers logit(Austin) - logit(Sacramento) on Texas-city prompts, by more than size-matched controls, without changing California or unrelated prompts.

**Corpus.** 20 new Texas/California city pairs (V01-V20) and 4 new unrelated prompts (M01-M04). No city overlaps corpus v1. The list was written without running the model on it; some names are ambiguous on purpose not being checked.

**Definitions.** D = logit(Austin) - logit(Sacramento). A = D(baseline) - D(target ablated). For control k, C_k = D(baseline) - D(control k), Q_k = A - C_k.

**Validity gate (baseline only, no interventions).** A pair is valid if D >= 0.5 on its Texas prompt and -D >= 0.5 on its California prompt. Invalid pairs are listed and left out of the statistics. Fewer than 16 valid pairs gives INCOMPLETE.

**Activity.** The feature is active on a prompt if its coefficient is > 1e-6. Activity rate over valid Texas prompts is reported as an outcome. Inactive pairs stay in the statistics with A = 0 and all Q = 0.

**Controls** (all chosen from the Texas baseline before any edit; applied at the same layer output and position):
- R, random: standard normal from PCG64 seed 20261119 + pair index, projected orthogonal to the target decoder direction, scaled to the target's decoded-contribution norm.
- M1, nearest feature, natural ablation: the active feature other than 15589 whose decoded-contribution norm is closest to the target's; ablate it. The norm ratio is reported.
- M2, nearest feature, size-matched: subtract that same feature's unit decoder direction scaled to the target's decoded-contribution norm.

M1 and M2 replace the v1 matched-feature rule (coefficient and norm both within 20%), which resolved on only 2 of 16 v1 prompts. No annotation-based exclusion is applied (no annotation file is pinned).

**Pass criteria, all required, over valid pairs:**
1. median A >= 0.5.
2. For each of R, M1, M2: median Q >= 0.25 and Q > 0 on at least 75% of valid pairs.
3. Specificity: |change in D| <= 0.1 on every valid California prompt, and |change in answer-minus-distractor| <= 0.1 on every M prompt.
4. Zero edit reproduces baseline logits exactly (max abs difference 0).

**Verdict.** PASS if all hold; FAIL if any of 1-3 fails with at least 16 valid pairs; INCOMPLETE otherwise.

**Repeat.** Two GPU runs must give identical result files. One CPU fp32 run is compared; the verdict must match and per-pair A must agree within 0.01.

**Also run, labelled after-the-fact:** the same script on corpus v1. Its controls were redesigned after v1 effects were seen, so v1 is supporting evidence only.
