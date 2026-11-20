# Research goal: independently checkable feature-level causal evidence

Status: charter established; implementation, execution and research outcome are not complete. The coordinator reports that a native goal is active. This document defines its acceptance; it is not runtime admission. Prepared 2026-11-20 for `Jimmycarroll2021/model-dissect` at base commit `e3b681b66f86dd11584b26f9da37df3b528cf2e4`.

## Goal and intended value

Deliver a reproducible, independently verified feature-level causal-validation toolkit, starting with the already frozen Gemma-2-2B layer20/source-local feature15589 case. A researcher should be able to inspect exactly what was changed, distinguish model fidelity from replacement approximation, recompute every claimed result from retained evidence, and repeat the admitted experiment without trusting an attribution graph or its feature label. Deliver reusable code, one fully adjudicated frozen case, and an accessible evidence report.

The intended societal contribution is more accountable interpretation of language models: make unsupported causal stories easier to detect and valid negative findings easier to retain and reuse. This is a research tool and case study, not evidence of safer deployed systems, a diagnostic for consequential decisions, or comprehensive model understanding. “Best in class” is an aspiration translated into the acceptance gates below, not a ranking or novelty claim. No literature-wide research gap is asserted. The existing published graph makes this first case replication/integration; no claim of first validation is made.

## Immutable starting point

The authoritative scientific design remains [selected-claim-v1.md](../specs/004-feature-attribution-validation/selected-claim-v1.md), together with [corpus-v1.json](../specs/004-feature-attribution-validation/corpus-v1.json). The [freeze manifest](../evidence/feature-attribution-pilot/protocol/frozen-design-manifest-v1.json), [independent design review](../evidence/feature-attribution-pilot/review/final-review-v1.md) and [closure overlay](../evidence/feature-attribution-pilot/protocol/closure-receipt-v1.json) establish design acceptance with execution blocked. The overlay supersedes the historical pending-acceptance field without editing frozen bytes.

The discovery node is `20_15589_8`; namespace `20-gemmascope-transcoder-16k`, layer20, local index15589. Exported121843834 is a different encoded identifier. Its geographic annotation and graph edge are hypotheses/discovery evidence. Exact deployed model, tokenizer and transcoder identities are still unpinned; a compatible executable feature adapter is missing. The graph's replacement/completeness scores do not establish any required numerical gate.

Preserve the prior three-model evidence, including Pythia's strict cached/full failure and GPT-2's qualified REFUTED finding. These are different experiments and are not baselines for ranking feature effects.

## Measurable definition of done

All requirements are conjunctive. A positive scientific finding is not required.

| ID | Required delivered evidence |
|---|---|
| G1 Identity and exposure | Append-only binding packet pins model/config/tokenizer/transcoder/dictionary metadata/runtime revisions and file hashes, hook conventions, input/answer token IDs and exact corpus; verifies every frozen input is compatible. Exposure ledger separates discovery, engineering fixtures and held-out evaluation. Any identity mismatch blocks admission. |
| G2 Reusable implementation | A documented adapter contract separates original execution, coefficient/decoded contribution, replacement approximation and evidence writer from model-specific loading. One real Gemma adapter supports the frozen decoded-delta edit with original residual preserved and ordinary downstream computation. A synthetic adapter/fixture exercises this interface, but is labelled synthetic and does not count as another validated real model. |
| G3 Numerical and edit validity | Independently checked raw evidence passes every original/editable fidelity, replacement reconstruction/output, locality, no-op, identity, technical rescue, restoration and weight-preservation requirement in the frozen design. Negative tests show rejection of wrong identity, wrong site/position, missing capture, invalid control, nonfinite result and resource exhaustion. These tests cannot substitute for real target gates. |
| G4 Completed first experiment | Execute every required arm for all16 Texas/California pairs and four capability prompts, within the admitted frozen limits. Save all per-item results and failures. Meet every eligibility/validity gate, then report SUPPORTED or REFUTED by the unchanged all-of criteria. No selective fixture removal, widening controls, post-effect threshold changes or favorable-seed retries. |
| G5 Independent reproduction | A separate executor actually repeats every required arm on the exact candidate and bindings, with separate comparison/control-audit code recomputing from lossless arrays; agreement is within the frozen tolerance and the eligibility/verdict matches. Summary comparison or implementer self-testing is insufficient. |
| G6 Reusable handoff | Deliver versioned code/environment specification, schema and raw-evidence manifest, one-command evidence audit and explicit commands for the admitted run, measured resource/cost accounting, failure/recovery instructions, limitations and a report mapping every claim to evidence. Independent verifier follows the documented route in a clean working directory, checks manifest integrity and reproduces the verdict. Broken/missing required files fail acceptance. |
| G7 Delivery | Integrate the exact independently accepted candidate through the authorised repository workflow and verify the delivered revision. Prepare a distributable, licence/provenance-reviewed research bundle while preserving current private visibility; public publication requires its existing authority or a separate explicit instruction. A private delivery is not public dissemination or demonstrated societal impact. |

A complete, valid and independently repeated REFUTED outcome satisfies G4: it answers the frozen threshold claim even if an effect is zero or smaller than required. Missing identities, impossible resource limits, failed approximation/eligibility, missing raw data or absent independent repeat are INCOMPLETE. Such failures are valuable feasibility findings but do not satisfy G4/G5 or complete this goal. Save the infeasibility result and escalate a prospective scope decision; do not redefine success. A new design must preserve this version, identify amendments and use fresh held-out material where prior exposure requires it.

## Baselines and comparison scope

The primary comparison is the pinned original model's Austin-minus-Sacramento contrast before and after the defined final-token decoded-feature ablation. Matched-feature and norm-matched random-direction controls test whether the target exceeds those perturbations; California and capability cases test only the frozen narrow specificity claim. Cross-prompt patching is secondary. Technical restoration is not semantic rescue.

Separately compare native and editable original executions for fidelity, and original versus replacement for approximation error. Neither comparison substitutes for the other. Reuse all numerical criteria from the frozen design unchanged: target median decrease>=0.5 logit; advantage>=0.25 and positive on>=12/16 for each internal control; specificity<=0.1 per case. The selected claim is the complete authority for all gates and formulae; this summary introduces none.

The toolkit is evaluated by reproducibility, rejection of invalid evidence and usable handoff, not by beating other interpretability libraries. No speed, cost, universality or state-of-the-art superiority claim is allowed without a separately preregistered fair comparison. Another model, another feature, training and automated discovery are outside this goal.

## Staged admission and bounded resources

1. Read-only binding/compatibility assessment: identify obtainable exact artefacts, licences/access requirements and engine semantics; inventory existing software and current host resources without loading the target. Report incompatibility rather than substitute a plausible checkpoint. Produce a feasibility plan and independent review before acquisition, installation or target execution.
2. Implement and test the adapter and evidence contract using synthetic data, retaining a separately owned native oracle/comparison path. Freeze engineering-only fixtures and their limits before target calibration; never inspect held-out causal results to debug or select features.
3. After reviewed acquisition and target-calibration admission, measure a bounded engineering pilot: time, memory, storage, call/token accounting, raw serialization, restoration and approximation/fidelity. Derive full-run feasibility from measured cost. Engineering work has explicit separate caps and receipts and cannot silently consume or evade the frozen experimental caps.
4. Independent execution admission binds exact code, artefacts, corpus, environment, operation semantics, metrics, control construction, complete call/storage ledger and repeat reserve. Only then execute the frozen experiment and independent rerun, followed by independent integrated-delivery verification.

Frozen per-role experimental ceilings remain1,024calls,65,536input token positions,1,800seconds,24GiB owned RAM and1GiB raw output; two roles reserve double, with zero new hosted spend. Inputs max64tokens, batch1, no cache. These are ceilings, not allocations or a feasibility claim. Before any expensive stage, specify bounded acquisition bytes, disk free reserve, CPU/thread use, process ownership, calibration calls/tokens/seconds/RAM/output, retries and repeat reserve; independently review against live capacity and active workloads. Do not terminate unrelated work or clean storage to make a plan fit. Any incompatible binding/cap preserves the frozen design blocked; no silent limit increase or reduced required evidence.

Persist terminal receipts, source/candidate hashes, prior side effects, remaining budgets and the next action before resuming an interrupted packet. Failed attempts count in their applicable budgets. Retry only with a new diagnostic reason or evidence of a transient failure. A blocker stops its dependent stage while other authorised model-free work continues.

## Immediate next executable step and current limits

Assign a bounded read-only binding and feasibility packet: reconcile saved graph/model/transcoder provenance; determine whether exact artefacts and the frozen original-model operation can be obtained; enumerate missing software/interface capabilities; inspect current resource capacity; produce an itemised acquisition, engineering-calibration and full-run/repeat plan for independent review. No model inference, held-out scoring or graph regeneration is needed for this first packet. The charter-writing assignment performs none of them.

This charter does not establish adapter implementation, scientific support/refutation, resource admission, automatic recovery or public availability. The repository's private status was supplied by the coordinator as the last verified state; this author did not query live GitHub. Public release remains separate from private verified delivery. The active native goal remains open beyond charter acceptance.
