# Historical-prefix cache drift: independently repeated diagnosis

The diagnostic localizes the first observed difference to **historical-token layer-zero outputs**, upstream of layer-one K/V production. It does not identify the responsible layer-zero operation, repair cache/full agreement or establish the cause of the final logit failure.

The [complete matrix experiment](cache-shape-matrix-result.md) failed24of9,120 unchanged comparisons. This follow-up examined precisely one failing left-padded length-one fixture, `heldout/length-1-left`, and one passing sibling, `continuation/continuation-length-1-left`. Each fixture was executed by the unchanged candidate and independently implemented reference, baseline only. Per actor/fixture: prefill length4, four cached single-token appends, then full length8. Each role made24calls and processed64actual positions, with128reserved.

The [final independent review](../evidence/cache-shape-followup/review/independent-repeat-final-v1.json), SHA256 `615bd93217dcdb2b89a4973dc1e207d8daab6c517c68944e93e0fa7482c778d5`, records `PASS_INDEPENDENT_EXACT_DIAGNOSTIC_REPEAT` and scientific status `UPSTREAM_HISTORICAL_PREFIX_DRIFT_OBSERVED_NOT_FIXED`. It independently checked732saved tensors,24call metadata records and288token-comparison rows. Every call in each role reproduced the original matrix baseline boundary and complete-cache hashes, establishing observational fidelity for these calls.

## What was observed

Both actors and both fixtures have the same localization pattern:

- Historical physical slots5and6 already differ at layer-zero output. These are real tokens; masked slots0–2 remain exact.
- Layer-one normalized inputs, raw K/V projections, K after head normalization and transformed K differ at the same slots. Their direct inputs already differ, so this is not evidence of an equal-input layer-one projection or normalization defect.
- Current physical slot7 remains exact across required captured comparisons.
- Position IDs and actual observed cosine/sine operands/results remain exact.

The differing captured values remain within the original elementwise tolerance. A passing sibling shares the upstream drift, so drift alone does not explain why the other fixture's later logits fail. The responsible layer-zero suboperation, Q/attention behavior, downstream amplification and final-logit causality were not isolated.

The full pass recomputes historical states at a different sequence width, while the cache retains states computed earlier. Shape-dependent FP32 kernel/reduction rounding is a plausible explanation, not a demonstrated implementation defect. This bounded finding does not establish impossibility of meeting the original tolerance or justify relaxing it. Further operation-level research is optional follow-on scope after requirement review; it is not an automatic new experiment mandated by this report.

## Evidence and resource boundary

Producer result SHA256: `af1d79d7feede35f7ba463b868d94f13dd24c2f6825609d4ebd41687aaead135`. Independent result SHA256: `f9634c43bbd38c31cc4f0f01e0ac48fc844b4ece3233159b0f777012416bb5fb`. Complete JSON identities differ because runtime/resource observations differ; the independent review establishes exact scientific tensor/metadata/comparison agreement.

Raw per-call arrays and metadata are retained in `EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-cache-followup-001-20261118/runs/`, in producer and independent directories. [Frozen protocol](../evidence/cache-shape-followup/protocol-v2.md) and [exact admitted candidate](../evidence/cache-shape-followup/runtime-candidate-manifest-v2.json) bind the capture and guard. Original strides/layouts were saved before contiguous copies; exact comparisons include signed zero; required observations have finite checks.

Producer/independent wall times were5.6503/9.5304seconds. All prior failed charges remain:1,137,888 historical plus256new reservations equals1,138,144, with zero remaining and zero active rows at the final review. Stage size159,000,842bytes and free space65,552,404,480bytes are that review's completion snapshot, not live monitoring.

The producer conservative combined RSS high-water was1,408,581,632bytes. The independent guard reported rusage high-water6,993,481,728bytes while its `/proc` VmHWM was20,424KiB; this discrepancy is directly recorded and its cause remains unresolved. No subtraction or cap weakening was applied. Independent conservative guard-plus-child high-water8,378,810,368bytes remained below8,589,934,592bytes. Treat these scopes explicitly rather than claiming that8.38GB was simultaneously observed resident memory.

No repair, tolerance change, retry, training, download or GPU run occurred. **Original native R2 FAIL and full new-mode matrix FAIL both remain unchanged.** Separately authorized same-mode M2 fidelity is a different, narrower accepted comparison; this diagnosis neither revokes it nor upgrades it into cache/full equivalence. Whole-objective review remains separate.
