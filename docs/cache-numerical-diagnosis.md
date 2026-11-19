# Why cached and full-prefix execution diverge: independently repeated diagnosis

**Independent verdict: PASS_INDEPENDENT_DIAGNOSIS_REPEAT_NOT_FIXED.** One historical fixture reproduces the cache/full discrepancy in both the native model and the editable reconstruction. They agree bitwise with each other within each mode. The measurements localize an early difference to attention computation despite identical first-layer Q/K/V values, and support shape-dependent FP32 arithmetic as a contributor. They do not establish a repair or a unique cause. A separate run reproduced all 400 saved tensors exactly, and the verifier independently recomputed all 48 primitive schedules for each run.

The [producer receipt](../evidence/cache-diagnosis/producer-execution-v1.json), SHA256 `285dcf8a8eea93c9d947637c705d64b9d8ed4b271ecc08e613dd34c8338f6c5c`, binds the raw result (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-cache-operands-001-20261118/runs/cache-operands-producer-v1/result.json`; external or omitted historical artifact), SHA256 `c950fab6154821918b2cc7f981dd9009b1662cf2c9683fe4b67e6487e48f913c`. The [independent final receipt](../evidence/cache-diagnosis/review/independent-repeat-final-v1.json), SHA256 `c6e59c77278813c755385b56ee5f59aa9dc3fa5d2b056b48c2e09eaba5cd1309`, binds the repeat and saved-array audit. This report now awaits its separate exact-document review.

## What was compared

The diagnostic used one previously exposed historical fixture, two implementations and both full-prefix and cached final-step execution. It captured operands and internal boundaries and evaluated shape-controlled primitive operations. The native and reconstructed implementations matched exactly at all reported cross-implementation boundaries in each corresponding execution mode. Thus this fixture’s discrepancy is not evidence that only the reconstruction implements the observed behavior incorrectly.

Within either implementation, full-prefix versus cached execution differed as follows:

| Boundary | Maximum absolute difference | Elements exceeding unchanged tolerance |
|---|---:|---:|
| Embedding | 0 | 0 |
| Recorded block boundary | 5.960464477539062e-7 | 0 |
| Final normalized representation | 1.4722347259521484e-5 | 0 |
| Vocabulary logits | 3.0174851417541504e-5 | 7 |

The maximum logit tolerance ratio was **1.4666059017181396**. A ratio above 1 is a failure under the original comparator; the tolerance was not relaxed. The seven failed vocabulary indices were 322, 3834, 4757, 4815, 5050, 5395 and 6037. The historical full R2 acceptance remains FAIL.

## What the operands reveal

Layer 0 Q, K and V were exact across modes, yet its unprojected attention output differed by up to **8.046627044677734e-7**. This establishes a difference after those identical inputs, rather than requiring a first-layer token/index mismatch to explain it. Across all 16 implementation-layer combinations, the shape-matched primitive reconstructions were exact. The single-query primitive contrasts reached a maximum absolute difference of **4.76837158203125e-6**. The producer performed 48 primitive contracts comprising 144 tensor operations.

These observations support the narrower diagnosis that changing computation shape can change FP32 contraction results, and that such differences occur before the final logit failures. They do not prove that this is the sole source of every historical mismatch, identify an implementation backend’s undocumented kernel internals, or establish that a proposed replacement attention computation preserves all required behavior. All captured compared operands were within the unchanged tolerance, while final logits nevertheless had seven failures.

The raw operand archive is all-operands.pt (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-cache-operands-001-20261118/runs/cache-operands-producer-v1/all-operands.pt`; external or omitted historical artifact), SHA256 `df9d35d20f7f5480475067b42f66c6d7ddf8df4786cc4ed24a298498d955e998`. Native/candidate full and cached boundary files live alongside it. These files are diagnostic outputs, not replacements for the immutable original checkpoint. This document does not load them or execute a model.

## Accounting, verification and next step

Each run executed **10 model calls / 44 actual positions**, reserving **256 positions**. Together they used **20 calls / 88 actual positions**, with **512 reserved**. Confirmed lifetime reservations are **1,036,432** under the approved **1,038,544** ceiling, leaving **2,112**. The independent receipt records zero active rows at completion; this is a snapshot, not ongoing monitoring. All six historical chains and admitted artifacts remained unchanged.

Phase start **1794993224.4247766** and deadline **1794995024.4247766** remain fixed. Six historical ledger chains and the original checkpoint file were unchanged. In-memory parameter rehashing was not instrumented; the reviewed probe has no parameter writes, but that is not the same evidence as a before/after in-memory hash check.

Independent raw auditing and repetition are complete. `/root/inference_diagnosis` is preparing a model-free repair proposal. The next useful candidate is to hold the attention contraction schedule consistent across cached and full-prefix execution, then test whether the final logit discrepancy disappears while preserving reference behavior. That is a proposal, not a demonstrated repair; its implementation choice, full acceptance coverage and costs still require exact review. No numerical fix or new model experiment is authorized by this report. Preserve the measured failure and original thresholds, including cases beyond this exposed fixture.

The independently repeated head-mediation finding and its [final report](echo-mechanism-result.md) remain valid within their tested same-mode scope; [the workbench](reverse-engineering-workbench.md) links the static map and causal evidence. This diagnosis advances the explanation of a remaining limitation. It does not complete the broader reverse-engineering goal.
