# Matching cached attention shapes removes the discrepancy on one fixture

A new, versioned numerical mode now passes the previously failing exposed fixture, and a separate execution reproduced all 800 saved tensors exactly. The verdict is **PASS_INDEPENDENT_REPEAT_LIMITED_SHAPE_FIXTURE**. This is a limited repair result for one unpadded length-8 prompt followed by three fixed single-token appends, ending at length 11. It is not full original R2 acceptance, and the untouched native cached implementation retains its historical FAIL.

The [independent final receipt](../evidence/cache-shape/review/independent-repeat-final-v1.json), SHA256 `7116e8136467817ae65dfad865b620da0fa89b66cc50bdb3050177e086ddc758`, binds the [candidate manifest](../evidence/cache-shape/candidate-manifest-v2.json), producer and independent raw results. This readable report is a separate final-document candidate awaiting review.

## The numerical change

The [preceding operand diagnosis](cache-numerical-diagnosis.md) showed that native and reconstructed implementations agreed within each execution mode, yet their full-prefix and cached executions diverged. First-layer Q/K/V were identical while attention already differed; shape-matched primitive contractions reproduced the full computation exactly. This motivated changing the shape of the cached attention calculation, without changing full-prefill computation or weakening the comparison threshold.

For a total key length T and one newly appended query, ordinary cached attention computes a single query row against T keys. The new policy places the actual rotated query in the last row of a T-row query array, fills preceding dummy query rows with zeros, computes the same FP32 QKᵀ, masking, softmax and value contraction, and retains only the final output row. Thus the retained query is evaluated with the full-width contraction shape. The dummy rows are discarded; their outputs do not become model states.

**The KV cache is genuine.** The previous K/V arrays are reused, with only the new token’s K/V concatenated. Projections, normalization and MLP run only for new tokens. Old hidden states, queries and MLP outputs are not recomputed. The changed cached policy was active from the first append. Full-prefill computation remains the original path.

Candidate and native-reference policies were authored separately. The reference temporarily changes the attention instances, delegates prefill to the original implementation and restores the exact original bindings in `finally`. This experiment is a versioned numerical alternative, not an edit to the original source or retrospective correction of its failed evidence.

## What was actually checked

Each role ran 12 calls and processed 63 positions:

- Candidate: prefill 8, cached 1+1+1, then full 11 — 22 positions, 5 calls.
- Independently authored reference: the same 22 positions, 5 calls.
- Untouched native baseline: prefill 8 and full 11 — 19 positions, 2 calls.

The [frozen protocol](../evidence/cache-shape/protocol-v1.md) retains its historical NOT_RUN header; subsequent execution receipts establish the actual result.

| Check | Independent result |
|---|---|
| Saved tensors reproduced across executions | All 800 byte-exact |
| Boundary comparisons | All 44 recomputed and byte-exact |
| Final-step per-layer operand comparisons | All 112 recomputed and byte-exact |
| Returned cached K/V prefixes | All 96 independently byte-compared and preserved |
| Input-cache preservation | All 96 source-reviewed live pre-call-clone assertions passed |
| Reference lifecycle | 24 cached attention calls, 16 delegated prefill calls, original bindings restored |
| Original artifacts and histories | Seven ledger chains, admitted files and producer outputs unchanged |

Both versioned policies reproduce the untouched native prefill-8 and full-11 boundaries exactly. They agree with each other at prefill, full execution and each of the three cached steps. At the final cached step, they match their own full-11 final-position boundaries and all recorded layer operands exactly. The preceding seven logit failures are absent on this fixture.

There was **no separate full-9 or full-10 comparison**. Per-step candidate/reference agreement at those intermediate steps is different evidence from cached-versus-full agreement at the final step. Input-cache checks were live assertions in reviewed code; returned-prefix equality was separately recomputed from saved arrays. Checkpoint files were unchanged; in-memory parameter hashes were not instrumented.

Raw outputs are retained in the producer result (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-cache-shape-001-20261118/runs/cache-shape-producer-v1/result.json`; external or omitted historical artifact), SHA256 `44099a1030f33763d6f24b5ec4dd672fd0eaef74245c4d98371eca4fac5a9c50`, and independent result (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-cache-shape-001-20261118/runs/cache-shape-independent-v1/result.json`; external or omitted historical artifact), SHA256 `784b933f5c3896e524797c40842d8163817a2cdde20aeb9adf9702ce1eedd01f`. Their neighboring tensor files contain the retained boundaries, layer operands and caches.

## Costs and limits

The two runs processed **126 actual positions in 24 model calls**, reserving **256 positions**. Confirmed lifetime reservations are **1,036,688**, leaving **1,856** under the approved **1,038,544** ceiling. All prior failed charges remain included. Phase start **1794994008.5799873** and deadline **1794995808.5799873** are fixed. The independent receipt records zero active rows at completion.

Matching query shape introduces dummy attention work; this study makes no speed, memory-efficiency or production-readiness claim. It tested one historical, exposed, unpadded fixture on the specified CPU FP32 backend, not padding variants, longer contexts, new prompts, or multi-token cached append batches. The original tolerance was unchanged, and byte equality was stronger than its numerical limit on this fixture.

## Next acceptance work

A full-matrix candidate is being prepared to cover both historical fixture sets: 24 distinct cases, all three baseline/no-op/restored arms, and native full-prefill fidelity in all arms. Preliminary accounting is approximately **50,526 positions per role / 101,052 for producer plus independent repeat**, subject to exact independent review. This exceeds the remaining allowance. A proposed larger phase disk allowance for complete raw tensors is also under review; no budget or disk-cap change has been approved by this report, and no new full-matrix run has occurred.

The next step is to finish the concrete full-matrix implementation, independently check coverage and resource bounds, and present any required approval against that exact candidate. A minimal passing example cannot waive the original full acceptance. The bounded head-mediation result remains independently supported in its tested scope; the broader reverse-engineering goal remains active.
