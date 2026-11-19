# Feature-level causal protocol — UNFROZEN template v0

Status: WAITING_FOR_GRAPH. This is a protocol structure, not a preregistered claim. No selected feature, numerical effect threshold, evaluation score or causal evidence is asserted.

## 1. Immutable identity and exposure record

Populate every item before freeze; an unknown is a blocker, never an inferred default:

- Graph canonical URL/ID, raw export SHA256, retrieval time, generation time when supplied; graph tool/version, pruning/edge/node thresholds, selected output and approximation settings.
- Original model repository, exact revision, configuration and weight file digests; tokenizer repository/revision/file digests, normalization and special-token behavior.
- Replacement model or transcoder/SAE repository, exact release/revision and weight digests; architecture, hook boundary, activation function and dictionary normalization convention.
- Selected layer (state indexing convention), feature index and full namespace; graph node ID; attribution score recorded only as discovery evidence.
- Exact discovery prompt bytes/hash and token IDs, BOS/EOS/chat template, masks/positions, feature token indices and output-readout index. Prompt labels are insufficient.
- Exact target logit token ID and comparator token ID if a contrast is used; one fixed scoring rule for every evaluation item.
- Environment/runtime/backend/dtype, inference mode, package/source identities; hosted access and export limitations.
- Exposure ledger: all graphs/prompts inspected, why this feature was selected, and every score already seen. Published graph examples are discovery data.

## 2. One falsifiable claim

Fill one sentence after discovery: “For [frozen prompt family/corpus], changing [one identified feature at fixed positions] by [specified ablation/patch operation] changes [fixed output contrast] in [direction] by at least [numeric minimum], exceeds [matched controls] by [numeric minimum], and satisfies [rescue/specificity requirements].”

This tests a feature contribution under a specified edit. It does not identify the complete circuit, prove the feature's label, or show necessity/sufficiency for all instances. A hosted attribution edge motivates the claim; graph magnitude is not a causal threshold.

## 3. Data and selection lock

Separate discovery prompt(s), engineering-only fidelity fixtures, and the frozen held-out corpus. Record exact texts/IDs/hashes, corpus size, tokenization constraints, contrast, edit-position rule, exclusions and minimum eligible count. Use a dedicated non-target capability set for specificity. All fixtures and controls must be selected without intervention effects or held-out scores. Baseline eligibility cannot silently remove unfavorable cases: freeze exclusions and minimum count, retain every exclusion, and classify inadequate data INCOMPLETE.

Do not reuse the discovery prompt as held-out confirmation. A one-prompt observation is a case study, not held-out generalization. No effect probe, donor patch or held-out score may run until the complete protocol and independent preregistration review are preserved. Engineering fidelity measurements, if later authorized, must have separate fixtures, predeclared tolerances and a logged prohibition on causal selection.

## 4. Faithful interception gates (all mandatory)

Distinguish three numerical questions; report each separately:

1. **Original execution fidelity:** the editable original-model execution versus its pinned native reference at declared boundaries and full output logits; fixed absolute/relative tolerances, argmax/contrast checks and fixture coverage.
2. **Replacement approximation:** original versus transcoder/replacement execution, reporting per-layer reconstruction residual norms (absolute and relative with a frozen denominator floor), output-logit discrepancy, distribution divergence and target-contrast discrepancy. Exact original fidelity does not prove approximation fidelity. Freeze acceptable bounds before measuring these fixtures.
3. **Intervention correctness:** capture feature coefficients and decoder contributions before/after the edit; verify only the admitted feature/positions change at the edit boundary. Later features may change through recomputation. Verify the untouched complement, exactly-once hook, finite values, disabled-edit baseline, identity/no-op patch, removal/restoration and weight preservation.

Specify error treatment mathematically: if editing the original residual by `(a_new - a_old) * decoder_feature`, retain the unedited reconstruction residual rather than recomputing an error term that cancels the perturbation. If editing a replacement network instead, define precisely which original computations are replaced and which nonlinearities/error terms are frozen. An effect measured only in the replacement supports a replacement-model claim; transfer to the original requires the same intervention to be validated there. No raw captures or incapable hosted API means INCOMPLETE for these gates.

## 5. Arms and controls

Freeze operations and required comparisons before scores:

- Unedited original/reference and replacement baselines, disabled-edit, identity patch and restored execution.
- Target-feature ablation: set the identified coefficient to zero at exactly the admitted positions; explicitly distinguish this from deleting an entire neuron/head/residual vector.
- Donor patch: replace that coefficient with its recorded coefficient from a prespecified clean/corrupt paired prompt, with a frozen position alignment rule. Do not replace the full hidden vector.
- Rescue: restore the recorded baseline coefficient after ablation and, where the claim concerns semantic transfer, include the prespecified clean-donor rescue. Define whether restoration is a technical check or a primary scientific criterion; simple restoration alone is not independent semantic evidence.
- Matched non-target feature controls: freeze eligible pool and deterministic selection based on layer/site, coefficient magnitude and decoder contribution norm; record matching tolerances and what happens if no match exists. No effect-based matching.
- Seeded random-direction control at the same boundary and positions, norm-matched to the decoded target perturbation, where that residual edit is valid. State that off-dictionary noise tests perturbation magnitude, not another semantic feature.
- Non-target capability prompts: predeclared contrasts/metrics and maximum allowed degradation, with the same intervention schedule where applicable.

For any arm judged inapplicable, freeze the reason and a replacement control before execution; do not omit it after seeing results.

## 6. Metrics and numeric freeze table

Prefer an absolute paired target-versus-comparator logit difference as primary metric when it answers the claim. Define the sign explicitly. If normalized recovery is primary, freeze its baseline denominator and strictly positive eligibility floor. Retain absolute changes as well so a tiny denominator cannot inflate evidence.

Before freeze fill numeric values for: corpus sizes; activation eligibility; denominator floor; native fidelity tolerances; reconstruction and output approximation ceilings; target effect minimum; target-minus-control minimum; direction consistency count; rescue tolerance/minimum; non-target degradation maximum; repeat agreement tolerance; confidence procedure/seed/resample count if used; all budgets. Distinguish descriptive intervals from decision criteria. Do not borrow prior head thresholds merely because they exist.

Set scientific minima from the smallest effect worth the bounded claim, on an interpretable logit/contrast scale; set numerical gates from the declared precision and faithful-operation requirement. Record the rationale without inspecting intervention effects. Freeze an all-required-criteria decision table; secondary metrics cannot rescue failed primaries. If uncertainty rules are used for the verdict, specify them now.

## 7. Verdict and independent repeat

- **SUPPORTED**: all identity, data, numerical, control and resource validity gates pass, all frozen primary scientific criteria pass, and an independent exact-candidate repeat reproduces the decision within the frozen tolerance.
- **REFUTED**: all validity gates and required repeat pass, but at least one primary scientific criterion fails. This means the scoped threshold claim failed, not that the feature has zero effect or no function.
- **INCOMPLETE**: any validity gate, eligibility count, identity, raw evidence, required arm, repeat or admitted resource condition is missing/fails. Failure to execute is not refutation.

The independent executor receives original requirements, immutable protocol/corpus/implementation manifests and retained data. It recomputes metrics/control matching/locality from raw arrays with separately authored comparisons and actually repeats every required arm. Summary JSON comparison alone is insufficient. Disagreement is INCOMPLETE pending diagnosis; do not average incompatible runs into acceptance.

## 8. Freeze, resources and recovery

Freeze package: protocol, graph export, identity manifest, corpus/token arrays, control selection, source/environment identity, budget and reviewer receipt; hash every artifact with UTC freeze time. Preserve an exposure ledger proving freeze preceded the first intervention and held-out scoring. The current drafting assignment provides no run admission.

Before execution set explicit limits for calls, token positions, wall time, memory, raw-output bytes, hosted charges (zero unless separately authorized), retries and the independent repeat reserve. Capture per-call terminal receipts, raw logits, feature coefficients and edit vectors, reconstruction residuals, masks/positions, exact arm identities and source links. Stop on invalid values, unexpected runtime identity, intervention-locality failure or budget breach. Preserve failures and partial results. Resume only after reconciling prior side effects and remaining budget. Amendments preserve earlier versions; exposed data cannot be relabeled unseen.
