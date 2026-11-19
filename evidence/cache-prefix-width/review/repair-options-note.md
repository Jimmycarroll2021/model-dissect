# Bounded algorithmic repair options after prefix-width localization

Read-only analysis; no implementation, primitive/model calls, gate changes or repair admission. The original native reference, FP32 comparator, genuine-cache preservation, fixture coverage and independent repeat remain required. Current full-matrix negative and narrower diagnostic findings are retained.

The candidate/native full paths perform row-batched linear projections, head normalization/RoPE, QK contraction, causal and finite padding masks, softmax, weighted V, output projection and MLP. The present cached mode only expands attention query working width. It still executes token-local projections/normalizations/output projection/MLP on a single new row; historical K/V are concatenated unchanged. Consequently query-width matching alone cannot ensure that a historical block output agrees with a later, wider full pass.

## What can and cannot be inferred

If one historical token has two different full-pass K/V byte patterns at widths6 and8, a single immutable cached copy cannot be byte-identical to both. This is a conditional statement about simultaneous exact cache equality, not proof the actual R2 gates are impossible: those gates concern bounded-error representations/logits and exact top1, not universal cache/full bit equality. The cache is required to preserve its own old bytes. It can differ slightly from a later full cache yet still pass every output tolerance.

No observed drift, one failed tolerance ratio, or an unchanged small fixture establishes that no compatible repair exists. Conversely, mathematical equivalence in real arithmetic does not imply FP32 gate preservation. Untouched-native fidelity must be measured with its existing comparator; it cannot be inferred from algebra or common candidate/policy-reference agreement.

## Branches after the fixed discriminator

| Localization | Smallest plausible candidate | Constraint and remaining uncertainty |
|---|---|---|
| First mismatch is a genuine input/layout/mask/index error | Correct precisely that error without changing arithmetic elsewhere. | Strongest prospect of preserving native behavior. Existing evidence has not established such an error. Never remove finite-masked padding keys. |
| Direct operands match; first mismatch is current-row Q/K/V projection, output projection or MLP linear shape | Execute only that identified current-row operation using an explicitly fixed row geometry/layout, padding dummy work and extracting the real row, while keeping stored K/V unchanged. | Can match a declared current full-pass geometry; it does not make previously stored states agree with every future width. Must not replay historical transformer states. If full paths stay untouched, this is a limited rounding-alignment candidate, not a general prefix-stability proof. |
| Direct operands match; QK or weighted-V contraction first differs | Align the identified contraction layout/work dimensions; preserve real operands, head order, scaling, causal and padding masks. | Current query-width mode already covers part of this idea and failed the complete matrix. Only a newly localized, materially different discrepancy justifies another candidate; repeating the same padding is not a repair. |
| Common scores match; softmax differs with extra causal-masked future columns | Consider a causal-prefix-only reduction schedule for each real query, retaining every finite-masked padding key. | To prevent future full-prefix drift, the corresponding full path must use the same schedule. That changes full FP32 arithmetic; unchanged-native fidelity remains an empirical gate. Prior canonical-attention candidate failed its native-prefill gate, so do not re-promote it unchanged. |
| First mismatch is normalization, activation, residual addition or an MLP boundary | Align that exact operation's shape/layout/ordering only when equal direct operands establish the contrast. | Do not blame a later differing operation when its input already differs. A cached-only adjustment cannot generally stabilize a native full operation whose result depends on future width. |
| All observed width6/8 operations match at token5 | Keep the discriminator negative; investigate the still-distinct cached-single-row versus full-row execution contrast only through a separately bounded decision. | Do not infer that wider capture or a generic backend change is automatically justified. |

## If the observed defect is native full-prefix numerical instability

A principled stable-cache design would give each token a future-width-independent projection and attention schedule from its first evaluation. Examples are row-fixed token-local operations plus causal-prefix attention reductions, or one explicitly fixed padded work geometry. Both preserve mathematical causal dependencies and immutable cached K/V. Both alter full-path FP32 arithmetic, potentially at many operations; neither is an established minimal compatible repair. The original native implementation must remain untouched as the fidelity oracle, and any proposed candidate must still pass every original/native gate. A new policy reference is not a substitute for that oracle.

If the current accepted contract also prohibits any changed full-path numerical policy—not merely vendor edits or fidelity failures—these broader designs are outside that contract. Under that interpretation, restrict candidates to observed interface defects or current-query alignment and report the potential constraint conflict rather than silently widening authority. A static analysis cannot prove such restricted candidates incapable of passing the finite matrix.

Refreshing historical K/V, full-prefix recomputation at every append, storing alternate history versions to select a future-width reconstruction, higher precision, fixture-specific correction constants, threshold changes and omitted fixtures do not constitute the requested unchanged-contract cache repair. Do not use them to turn the gate green.

## Recommended controller decision after localization

Prefer exactly one locally supported change to an identified operation with equal-input evidence. Explain separately whether it leaves full arithmetic unchanged, preserves old K/V bytes, and retains untouched-native fidelity. If localization instead establishes native future-width instability, report that as the result and distinguish a measured candidate opportunity from the unproven general compatibility question. Reuse existing accepted gates; this note introduces no additional compulsory gate or experiment.

Sources inspected: frozen shape_cache_matrix_candidate.py; shape_cache_reference_matrix.py; oracle/vendor/model_minimind.py; oracle/adapter.py; historical evidence/continuation/protocol-v2-proposal.md. Historical canonical-attention rejection remains documented in the maintained project plan; it is a reason against unchanged retries, not a universal impossibility theorem.
