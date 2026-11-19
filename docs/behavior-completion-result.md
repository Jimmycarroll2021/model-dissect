# Completed-response diagnostic — independently repeated bounded negative

The first bounded continuation test produced **no completed answers within24 generated tokens**. All48 responses reached the cap without the pinned end-of-turn token. COPY, SUBTRACT and COMPARE therefore each scored0/16 under the frozen requirement for a correctly terminated answer. No family was selected and fresh qualification was not run.

This is a **length-censored result**, not48 demonstrated semantic errors. We now have the actual continuations beyond the first token, but still do not know whether these responses would eventually give correct answers. The answer parser did not reject completed explanations: every response was marked incomplete before answer parsing. Consequently this experiment does not isolate formatting, task competence, termination behavior or checkpoint quality as the cause.

The investigative tools are usable: the unchanged reconstructed computation generated deterministic continuations, validated the pinned prompt bytes/token IDs, recorded every step and preserved resource/source identity. A qualified behavior or internal mechanism has not been discovered. Completing this diagnostic does not complete the broader model-understanding objective.

## Observed responses

The model was asked to respond in Chinese. These are full decoded24-token continuations from the first case in each family; each is truncated by the experiment and has no EOS.

| Task and expected answer | Observed continuation at the cap |
|---|---|
| Copy: 甲 wrote0, 乙 wrote9, 丙 wrote6. Expected0. | `甲写下数字0，乙写下数字9，丙写下数字6。甲只写下` |
| Subtract:62−62. Expected0. | `我们知道，一个数减去它本身等于另一个数。所以，62减去62等于多少呢` |
| Compare:50 versus2. Expected大 (larger). | `这个问题的答案取决于你所指的“50比2大还是小”的定义。在数学中，50` |

The copy response echoes the given digits; the other examples begin explanatory text. This descriptive observation is not a diagnosis. Other cases remain available in the complete raw result; no examples or outputs were used to revise prompts, grammar, threshold or generation cap.

| Family | Correct terminated answers | Strict answer-only successes | Truncated/no EOS | Selected |
|---|---:|---:|---:|---|
| COPY |0/16|0/16|16/16|No|
| SUBTRACT |0/16|0/16|16/16|No|
| COMPARE |0/16|0/16|16/16|No|

The frozen selection gate was13/16 for the first eligible family. Each0/16 observation has a descriptive Wilson95% interval of0–19.36% for **success under this exact response cap/grammar/protocol**, not eventual semantic correctness or a population capability guarantee. These fixed reused screen inputs are selection evidence. Qualification would have required26/32 on previously unobserved inputs; it remains NOT_RUN. Parsed-wrong, unparsed-complete and control-token parser classifications are all0; incomplete is48. No interventions followed.

## Execution and independent evidence

Protocol: [frozen BEHAVIOR-002](../evidence/behavior-completion/protocol-v1.md). Exact pre-execution candidate: [manifest-v1](../evidence/behavior-completion/candidate-manifest-v1.json), SHA256 `f589ef6490bc27793efafb1aefe708d0d8f3244d058febbb18d6aec27f67601c`. The preparation manifest remains immutable; later task/status updates are separate delivery metadata.

The producer raw response result (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-behavior-002-20261118/runs/response-screen-v1.json`; external or omitted historical artifact), SHA256 `87c82eef89d6985f6334cecd3302e0a032210d7af7b0132a1295ca71a42c90cf`, contains all48 responses and1,152 steps. Per-step journal (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-behavior-002-20261118/runs/response-screen-v1.json.steps.jsonl`; external or omitted historical artifact), SHA256 `a57ede30c8601e039f693194a99ebd5bdb28117aea63304f86f15a479bb962bb`, was flushed and synced after steps/cases. Saved vocabulary logit hashes establish repeatable vector identities if matched; full vocabulary arrays were not retained.

[Producer execution receipt](../evidence/behavior-completion/screen-execution-v1.json) records54816 actual/reserved positions,49.11066387302708 wall seconds and85.455596 CPU seconds. Every forward recomputed its complete prefix with cache off on CPU FP32/two threads. Actual prompt lengths28–48 and maximum forward prefix71 stayed within the frozen interface. No original weight was edited. Both historical ledger hashes remained unchanged. The guard's conservative sum of independent child and guard highwater measurements was7758471168bytes, below8589934592; this is not model memory alone.

**Independent repeat: PASS_INDEPENDENT_NEGATIVE_REPLICATION.** The [independent final receipt](../evidence/behavior-completion/review/independent-repeat-final-v1.json) confirms all48 rows,1152 token IDs, selected logits, full-vocabulary logit hashes, stop reasons, parses and position records matched exactly. The entire per-step journal is byte-identical. The reviewer independently reconstructed all48 decoded texts from the byte-level vocabulary and recomputed the aggregates. Original and staged artifact identities were unchanged.

The independent raw result (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-behavior-002-20261118/runs/response-independent-v1.json`; external or omitted historical artifact) has SHA256 `17d70fd7f8ab4f8172c3f45b858feddc5b1d41bfc4478c88d16d6536b26e4e4e`. Its reviewed guard completed in47.03209758794401 wall seconds,84.77042399999999 CPU seconds, with conservative combined highwater7764418560bytes below8589934592. The producer's separate [raw audit](../evidence/behavior-completion/review/screen-raw-review-v1.json) and [screen admission](../evidence/behavior-completion/review/candidate-review-final-v1.json) preserve the preceding gates.

Across the two completed runs, actual input positions total109632 across2304 forwards,96.14276146097109 wall seconds and170.22602 CPU seconds. Reserved positions are54816 screen plus100512 independent =155328 this phase; the repeat's full reservation remains charged despite skipped qualification. Add historical295600 for450928 lifetime reserved positions. Model wall reservations total2400seconds. Reservation ceilings are not actual execution duration. Qualification is NOT_RUN_NO_SELECTED_FAMILY; causal work is outside scope. There are0active ledger rows, and no more model runs are admitted by this result.

Phase start1794985573.2121327 and deadline1794992773.2121327 remain fixed; no clock reset or refund occurred. Both historical ledger hashes remain unchanged. Native goal is null; no continuing process is implied. The bounded scientific diagnostic and independent repetition are complete. This final document is a separate delivery candidate for exact review; it does not complete the broader model-understanding objective.

## What remains to understand

Before another internal intervention, the next substantive decision is whether the frozen response window and pinned input mode are adequate for measuring this checkpoint's task behavior. This experiment cannot distinguish a response that merely needs more than24 tokens from a response that never reaches a correct answer. Any longer-window or format comparison requires its own explicit, frozen contrast, cap, resource admission and independent verification. It must not retroactively relabel this screen, cherry-pick prompts or change the old scoring rule.

Historical [weight dissection](weight-dissection-result.md) remains accepted with its narrow static-OV hypothesis REFUTED. Historical cache/full R2 FAIL and earlier mechanism INCONCLUSIVE remain unresolved. This phase adds observed generated responses; it does not repair those findings, explain a circuit, or establish broad model understanding.
