# Control v2 spec: a like-for-like random control (frozen before it is run)

Frozen 2026-11-24. This file and `control2.py` are committed before the first run.

**Why.** The independent re-measurement found the v1 random control is not like-for-like: the spec did not say whether the random vector is added or subtracted, and a random vector of the removed size makes the residual larger (lowering every logit through later norms) while removing a real feature usually makes it smaller. The "beat the largest of 16 draws" rule was also noisy. This revision is made after seeing the v1 results; it changes the control and the distinctness rule only. E, W and E / W are not touched.

**Population.** The same 75 edges, same prompts, same positions, same model and convention as `pilot-v1-spec.md` and `scale-v1-spec.md`.

**Gate.** The runner recomputes E for every edge and requires |E - stored E| <= 1e-3 against `results/pilot-v1-run1.json` and `results/scale-v1-run1.json`; otherwise it stops.

**Control.** Let v be the removed contribution (coefficient times decoder row), r the residual stream at the output of the edge's layer at the edge's position in the unedited run, r_hat = r / |r|, u the unit decoder direction. A control vector is

    c = a * r_hat + b * w,   a = v . r_hat,   b = sqrt(|v|^2 - a^2),

where w is a random unit vector (PCG64 standard normal draw, float32) made orthogonal to both r_hat and u. So |c| = |v| and c . r = v . r. The control is **subtracted** at the same place v is subtracted (output of post_feedforward_layernorm), so the edited residual r - c has exactly the same norm as r - v. 64 draws per edge, seeds 20262000 to 20262063. The control effect C_i is the drop in the target's mean-centred pre-cap logit, the same measure as E.

**Rule.** An edge is DISTINCT if E > 0, E - mean(C) >= 0.25, and at most 3 of the 64 draws have C_i >= E. Verdict v2: WRONG_SIGN if E <= 0; NOT_DISTINCT if not distinct; otherwise WEAKER (E / W < 0.5), HOLDS (0.5 to 2), STRONGER (> 2). The matched-feature controls of v1 are reported as they were and play no part in the v2 verdict.

**Reported.** Per edge: E, mean, largest and number-at-or-above of the 64 draws, norm check (| |r - c| - |r - v| |, largest over draws), v1 verdict, v2 verdict. Overall: verdict counts v1 and v2, and the edges whose verdict changes.

**Repeat.** Run twice on GPU; result files must be identical (cmp).

**Not claimed.** One prompt per edge. The norm is matched at the write point, not at every later layer. Pass marks (0.25, 3 of 64) are choices.
