# NUMERICS-003 proposed attention-policy amendment

2026-11-17. Status: PROPOSED, requires separate exact-script/protocol review before execution. No runs performed by this advisor. Source-assisted: upstream source, adapter, comparator and current specification were inspected.

## Evidence and precise unresolved cause

D3 result `evidence/continuation/diagnostics/D3-result.json` reports 48 identical-input Q/K/V projection checks exact under strict mode. Its earliest valid sampled first-block difference is the scaled attention-score tensor, max absolute 3.814697265625e-6. Complete prefix K operands were not checked, so this does **not yet prove** the score contraction itself caused the discrepancy. Later stale-local samples are excluded. The strict full suite still fails 234/2640 rows, worst ratio 2.108029842376709. These failures remain authoritative and R3 remains blocked.

## One additional diagnostic only

Amend the exhausted diagnostic allowance by exactly one process: CPU FP32, strict common backend, two threads, at most 4096 reserved input/derived-operation positions and 300 seconds, charged to the original ledger and deadline. No fresh elapsed, download, GPU or token budget. Reserve operation counts before execution; shorten captures if they exceed the allowance. Use only the already-exposed length8 unpadded case, continuation step2, and bounded synthetic mask fixtures. No thread/branch/device search.

1. Capture complete first-block Q-last, **all** K-prefix and V-prefix operands for cached and full execution, including head expansion and transpose. Test bit equality of operands, effective position IDs, scale and allowed-key sets before attributing differences to contractions. Compare cached-prefix preservation independently. If K/V differ, identify their first differing prefix row and stop contraction-only diagnosis; do not claim the cause isolated.
2. On identical captured operands compare native batched score contraction against independently implemented one-query-at-a-time contraction. Keep feature-axis reduction in FP32 and the same head grouping/scaling. Replay the same query with exactly the same operand layout/strides or canonical contiguous copies on both paths; record the choice. Compare intermediate scores before masking, masks, softmax and value contraction separately. This establishes which primitive/layout changes the result, not whole-model acceptance.
3. Test the proposed common policy for each query: process one query against its causal prefix, apply the original finite additive key-padding mask, softmax and V product, concatenate outputs in original order. The policy may remove only positions excluded by causal negative infinity. Do not remove finite-masked padding positions or change their additive value. For a later full-pass query and the corresponding cached query the prefix length, row shape and reduction dimensions must coincide.
4. Independently check synthetic attention against the unchanged native expression on identical **individual-query shapes**, including causal boundaries, grouped heads, finite padded keys and unequal values. Require the original tolerance/finiteness and exact indexing/mask equality. Count these primitive calls. This checks algebra/indexing without pretending native batched execution is bitwise invariant.

This diagnostic may instantiate no further full-model variant. Save operands' hashes/shapes, comparisons, query/cache counts and failures incrementally. Stop after this single run, whether conclusive or not.

## Conditional candidate and independent oracle policy

Only if the complete-operand and isolated-primitive checks support shape-sensitive attention arithmetic, propose one source-assisted execution candidate: canonical per-query causal-prefix **attention only**, in FP32, retaining all other operations and genuine KV cache. This is an ordinary scheduling change to the same real-arithmetic attention expression, but it changes floating-point operation order and therefore must be declared. It is not a recovered hidden mechanism or an automatically valid oracle.

Keep original checkpoint and vendor bytes unchanged. Candidate implements its own primitive composition. A separate oracle owner implements the declared attention execution policy around native reference layers; no candidate import, shared attention helper or global monkeypatch. Reference-layer replacement/adaptation must be explicit in protocol/manifests; it changes the previous declaration that all native layers execute unchanged. A verifier checks both implementations against the mathematical mask/index contract and unchanged native individual-query primitive behavior. Common-policy agreement alone never establishes correctness.

Do not silently call the policy-reference the unchanged-native model. Preserve three clearly labeled actors: unchanged native reference, policy reference, independent candidate. Preserve all historical failures. This source-assisted amendment changes numerical execution policy, not graph mathematics, dtype or tolerances.

## Existing native gate is mandatory; avoid inventing a stronger gate

The actual `oracle/protocol.json` requires native equivalence for contiguous unpadded positions. `oracle/check_adapter.py` implements this as **prefill** checks at lengths1,8,32,128 across embedding/block/final/logits, plus separate real-token padding invariance. It does not currently require policy output to match both historical cached and full outputs for every continuation. Preserve and rerun the actual native-prefill and padding-invariance checks at the original tolerances after an adapter policy change. Identical-query primitive checks are supplementary; they cannot replace these whole-model checks.

My earlier suggestion to require every changed-policy output to match both original modes was broader than the existing native-adapter smoke. Do not introduce that extra acceptance gate without a reason. Equally, do not delete the existing native-prefill gate. Record original-native cached/full comparisons as diagnostic evidence if budgeted; label them separately rather than claiming them passed.

A worst asymmetric tolerance ratio above2 alone does not prove mathematical infeasibility: compatibility depends on per-element tolerance intervals and which comparisons are actually required. Nevertheless if the new adapter fails the existing native-prefill check, the proposed remedy cannot satisfy the unchanged gates. Stop that candidate and report the concrete failed native-equivalence requirement. Reclassifying that gate as algebra-only would be a substantive acceptance change and is not admitted here.

## Freeze, evaluate, stop

After successful diagnostic and independent policy review, freeze one candidate with exact source, adapter, environment, comparator and disclosure identities. Freeze fresh private12case parity fixtures before its evaluation; the prior failed suite is exposed. Admit at most one combined acceptance run and one independent repeat, each at most18000 reserved tokens/600seconds, including native-prefill and padding checks in the precomputed reservation. If their combined workload exceeds18000, reduce no coverage: separately reserve the native gate within the same original lifetime cap and declare the exact additional amount before admission. No uncounted supplementary runs.

All existing R2 gates remain: common fixed FP32 settings, finite outputs, original elementwise atol/rtol, exact top1, all boundary sites, all lengths/paddings, four genuine cached continuations, full-sequence comparisons, no-op/restoration, unchanged originals. Keep at least100000 tokens for R3 and original four-hour deadline. Only successful complete R2 plus independent exact-candidate verification admits R3.

No automatic second algorithm/backend candidate follows failure. If operand mismatch falsifies the attention-only hypothesis, or native equivalence/full R2 fails, save that exact limitation for coordinator decision. This bounded amendment supplies a plausible next test, not a guaranteed means of satisfying the original numerical contract.
