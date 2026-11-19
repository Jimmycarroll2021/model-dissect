# A verified local mechanism for MiniMind’s echoed digit

Replacing the output of **attention head 2 in layer 6** transfers a substantial part of the model’s preference between two echoed digits. A separate execution reproduced the complete experiment exactly. The independently reviewed verdict is **SUPPORTED_BOUNDED_HEAD_MEDIATION**; the coordinator accepted this bounded finding. Layer and head numbers are zero-based.

The result concerns the first digit in the spontaneous response prefix “甲写下数字…” (“甲 wrote the digit…”). It does not establish final-answer accuracy, a complete copying circuit, the source token attended to, necessity or uniqueness, or an explanation of repeated text and EOS behavior. The broader reverse-engineering goal remains active.

[Independent actual-repeat review](../evidence/echo-mechanism/review/recovery-independent-final-v1.json), SHA256 `5dc101f6e006281d4419bba06f6a54cb080d310c8209b2bc72d19618d937f70b`, binds the exact candidate, runner, guard, policy and raw results. This readable report is a new delivery candidate awaiting separate document review.

## What changed inside the model

The frozen paired prompts differ only in the first person’s digit. Both naturally generate the same four-token prefix, followed by their respective digit; eligibility also requires a sufficiently separated clean/corrupt digit-logit margin. There are 8 development pairs and 16 fresh held-out pairs, with all fixed denominators retained.

At the final prefix position, zero-based position 51, the attention block has a concatenated 768-value vector: eight heads × 96 values. A prehook on layer 6’s `o_proj` clones its input and replaces **only head 2’s 96 values**, coordinates `[192:288]`, with the matched opposite prompt’s baseline values. All other positions and coordinates must remain unchanged. The experiment performs both donor directions, then compares a same-layer control head, no-op and restored runs. No weights are edited.

For row-vector convention, the output projection is `y = h W_Oᵀ`. The direct contribution of head 2 is `h₂ W_O[:,192:288]ᵀ`, a 768-value residual contribution. Replacing the head changes that immediate contribution by `(h₂_donor − h₂_receiver) W_O[:,192:288]ᵀ`. Subsequent residual, MLP, normalization and output computations propagate its effect to the digit logits. The saved trace contains the actual 96-value clean/corrupt head vectors and their 768-value projected contributions. This is a measured intervention at the receiver; it does not isolate an attention edge from any source position.

[The all-tensor computation guide](model-computation-map.md) explains where this equation fits among all 91 tensors. The [frozen scientific protocol](../evidence/echo-mechanism/protocol-v1.md) preserves selection, qualification, intervention and scoring rules; its original pre-execution status/resource paragraphs are historical. The recovery receipts bind the later approved accounting and unchanged science.

## A concrete held-out example

For `echo-08`, one prompt says that甲 wrote **3**,乙 wrote 0 and丙 wrote 4; its matched partner changes only甲’s digit to **9**. Both ask what甲 wrote. Their natural next output after “甲写下数字” is respectively 3 or 9.

Let `D = logit(3) − logit(9)`. The baseline values were **8.129873275756836** for the 3 prompt and **−6.80893611907959** for the 9 prompt. Moving head 2’s output from the 3 prompt into the 9 prompt increased D by **8.3811674118042**. The reverse transplant reduced D in the 3 prompt by **8.970104217529297**. The normalized recoveries were **0.5610331580173408** and **0.6004564340067019**, so this pair’s primary score—the smaller direction—was **0.5610331580173408**.

The same operation on the selected control head 5 had a minimum directional recovery of **0.0007660006778394198** on this pair. Its paired advantage was therefore **0.5602671573395014**. These are observed logits and normalized differences, not percentages of the model explained. Exact raw values and vectors are in `heldout[0]` of the producer result (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-echo-recovery-001-20261118/runs/echo-recovery-producer-v1.json`; external or omitted historical artifact), and independently reproduced in the repeat result (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-echo-recovery-001-20261118/runs/echo-recovery-independent-v1.json`; external or omitted historical artifact).

## Frozen scoring and held-out result

Writing clean/corrupt baseline gaps as Dc and Dk, the two directional scores are `(Dpatched_k − Dk)/(Dc − Dk)` and `(Dc − Dpatched_c)/(Dc − Dk)`. Each pair’s primary score is their minimum. This requires an effect in both directions; raw ratios are not clipped. Ineligible pairs remain in the fixed denominator with the protocol’s zero primary score.

All 64 heads were scanned on the 8 development pairs. The target was the largest median primary recovery; the control was the **weakest development-selected same-layer head** by the frozen rule. Thus head 5 is a deliberate weak control, not a random head or evidence that all other heads are unimportant. The selected target/control were frozen before held-out evaluation.

| Held-out measure | Observed | Frozen requirement |
|---|---:|---:|
| Eligible pairs | 16/16 | At least 14/16 |
| Positive movement in both directions | 16/16 | At least 13/16 |
| Median minimum directional recovery | 0.5534745383939315 | At least 0.25 |
| Median paired advantage over control | 0.5535074254291052 | At least 0.15 |
| Target minimum-recovery range | 0.4320575127490965–0.6506104042720143 | Descriptive |
| Combined no-op and restored full-vocabulary outputs | Exact | Exact required |

**55.35% means the median of each pair’s minimum bidirectional normalized digit-logit-gap recovery.** It is not 55% of the model, a circuit, accuracy or all copying behavior.

A post-hoc full-vocabulary top-1 check found that the target transplant made the donor digit highest on 9/16 and 11/16 cases in the two directions. The control did so on 0/16 in both directions, retaining the original digit on all 16. These counts are descriptive, not the primary preregistered threshold. The descriptive sign probability is 1/65536; shared templates and development-selected control prevent treating it as a calibrated random-population causal proof.

## What was independently repeated

The independent run repeated all **1,480 calls**, preserving every journal full-vector hash, prepared fixture, parity/instrumentation record, all 512 development scores, selection and all held-out traces. Native/reconstructed boundary tensors at prefix lengths 52 and 128 matched exactly between runs. No-op/restoration full-vocabulary vectors and before/after weight checks passed. All admission-bound files and five historical ledger chains remained unchanged.

The earlier failed run is retained: its 295 durable call artifacts and 78,112-position reservation were not deleted or refunded. Its failure was a serialization-view storage defect, not negative causal evidence. The reviewed owning-clone repair was exercised on all 1,480 saved producer vectors, each with a 25,600-byte storage payload. The scientific protocol and selection thresholds were unchanged after the failure.

Each successful run used **76,632 actual positions** and reserved **76,800**. Confirmed lifetime reservations are **1,035,920**, leaving **2,624** under the user-approved **1,038,544** ceiling. The independent receipt records zero active runs at completion; later authorized work has its own state. Failed and prior charges remain included. This report authorizes no additional inference.

The finding supplies a verified local causal component and real activation traces. It leaves source-edge attribution, necessity, uniqueness, broader behavior and the original cache/full numerical discrepancy unresolved. Earlier coordinate INCONCLUSIVE and static-OV REFUTED results retain their meanings.
