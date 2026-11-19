# Weight dissection result — A1–A6 accepted

The weight atlas is independently accepted for A1–A4. The controlled investigation and independent repeat have **PASS_A5_PROCEDURAL_AND_INDEPENDENT_REPEAT**, with scientific verdict **REFUTED**. A1–A6 are accepted through [final independent review](../evidence/atlas/final-review-v1.json) and [controller acceptance](../evidence/atlas/controller-acceptance.json). Native goal status is governed by the goal tool.

The verified practical finding is precise: the strongest tested static OV direction did not predict greater selective influence than its matched neighboring-head control on these frozen inputs. The target intervention changed the measured logits, but the control changed the selected contrast more. This is a useful negative result about one static cue and one intervention, not a claim that the selected head has no function.

## What is available

- Readable weight atlas (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/runs/atlas-v1/atlas.md`; external or omitted historical artifact)
- Full atlas JSON (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/runs/atlas-v1/atlas.json`; external or omitted historical artifact)
- [Independent A1–A4 acceptance](../evidence/atlas/atlas-acceptance-v1.json)
- [Frozen experiment protocol](../evidence/atlas/a5-protocol-v1.json)
- [Independent repeated-result review](../evidence/atlas/review-a5-independent-result-v1.json) and raw repeated result (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/runs/a5-independent.json`; external or omitted historical artifact)
- [Reviewed confirmation result](../evidence/atlas/review-a5-confirmation-result-v1.json)
- Raw confirmation result (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/runs/a5-confirmation.json`; external or omitted historical artifact) and per-arm journal (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/runs/a5-confirmation.json.arms.jsonl`; external or omitted historical artifact)
- Prepared/tokenized input receipt (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/runs/a5-confirmation.json.prepared.json`; external or omitted historical artifact)

The atlas covers91 stored tensors and68,827,392 stored elements; all measured values were finite. It includes16 sampled spectral matrix analyses,4 representative attention heads and16 OV direction views. Forty-two matrices were skipped under the spectral cap and33 tensors were nonmatrices. Independent mathematical checking reported359 checks with0 failures, including8 representative scalar-statistic rows,16 spectral analyses,4 geometry checks and16 OV directions. Architecture labels and computational conventions are disclosed as source-assisted/prior-informed; this work does not uniquely recover the original source from weights.

## The controlled test

The actual atlas selected layer7/query-head7/KV-head3, coordinate767 by maximum residual norm among its16 sampled OV directions. The frozen contrast uses token302, decoded `。`, versus token3070, decoded `ons`. Their spellings identify the measurement; neither is called a semantic feature.

The target arm halved that head's output-projection columns672:768. The control attenuated neighboring head6, columns576:672, with a matched weight-perturbation Frobenius norm. Its coefficient was0.45625391023826667; measured perturbation norms were3.3020963176667433 and3.302096144048426. Equal weight norms do not guarantee equal activation changes, so the control constrains one alternative explanation rather than all alternatives.

Sixteen development prompts and32 fresh confirmation prompts each ran baseline, no-op, target, matched control and restored arms. Execution was CPU FP32 full-prefill, cache off, without generated continuations. Reviewed validity includes exact baseline/no-op/restored equality at declared boundaries, restoration of the original working tensor object and unchanged original artifact identities. All interventions used temporary working copies; original checkpoint files were unchanged.

| Frozen confirmation criterion | Observed | Criterion |
|---|---:|---:|
| Mean absolute target log-odds change |0.1542705511674285|≥0.10|
| Mean target-minus-control contrast magnitude |−0.1860829326324165|≥0.05|
| Mean target-minus-control normalized selectivity |−0.7303070982739478|≥0.25|
| Cases where target contrast change exceeds control |11/32|≥24/32|

The frozen refutation rule is met: mean target advantage≤0 and wins≤16/32. The first magnitude gate passing does not rescue the failed comparative claim. Selectivity divides the contrast change by centered whole-vocabulary logit RMS; raw journals also preserve signed contrasts, KL and top1 changes. These operational thresholds are not a statistical-significance claim. Independent repetition reproduced all160 FP32 logit vectors bitwise; the160-arm journal and full preparation receipt are byte-identical to confirmation. Its reviewer completed692 additional checks and reused3111 unchanged confirmation checks, independently recomputing aggregates from repeated logits. The repeated selectivity aggregate was−0.7303070982739477 versus confirmation−0.7303070982739478, an aggregation-order rounding difference; the frozen verdict remains REFUTED.

## Using the delivered artifacts

Open the readable report first, then use the JSON for tensor-by-tensor values and the journals for individual experiment arms. The runtime implementations are atlas.py (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/src/minimind_re/atlas.py`; external or omitted historical artifact), atlas_probe.py (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/src/minimind_re/atlas_probe.py`; external or omitted historical artifact), forward.py (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/src/minimind_re/forward.py`; external or omitted historical artifact), and atlas_guard.py (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/runtime/atlas_guard.py`; external or omitted historical artifact). The original project is not a Git repository; candidate identity comes from exact file manifests.

The successful atlas execution's exact child CLI argument vector is saved in `guard_row.argv` of producer-execution.json (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/evidence/atlas/producer-execution.json`; external or omitted historical artifact), and its guard-bound parameters are in the frozen policy (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/runtime/atlas-policy.json`; external or omitted historical artifact). This is the corresponding historical guarded command shape:

```sh
cd EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118
/usr/bin/python runtime/atlas_guard.py   --workspace EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118   --policy runtime/atlas-policy.json   --policy-hash a8a2ab2d2138062cd266aac645f8d898e08f984ee9d46f1239c3f97523e4d21a   --admission evidence/atlas/admission-atlas-producer-v1.json   --run-id atlas-producer-v1 --kind static --positions 0 --timeout 300
```

The run ID is already consumed and is not a rerunnable quickstart. Read the saved results without executing another experiment. A future exact run requires a new reviewed admission and valid remaining budget; the CLI does not create that authority or an automatic reusable allowance. The displayed command reconstructs the guard interface; the receipt's actual child argument vector is authoritative for what ran.

## Evidence limits and preserved failures

Historical cache/full R2 remains FAIL. Earlier same-mode M2 evidence is narrower and reused by identity; this phase did not rerun a new parity suite. The earlier coordinate investigation remains INCONCLUSIVE and the earlier behavior screen remains STOP_NO_QUALIFIED_PAIR. None was relabelled by this phase.

The static atlas repeat produced complete matching output but its guarded process failed. That failure and its300second reservation are preserved. [The bounded output-reuse review](../evidence/atlas/review-repeat-evidence-v1.json) accepted the complete output separately; a subsequent guard v5 correction passed review. This is not a successful-process claim for the failed repeat.

All three A5 model slots are consumed. The phase reserved120000 positions and processed12750 actual positions; lifetime reservations including historical175600 are295600. Static reservations total900seconds and model reservations1800seconds. Reservations are ceilings charged by the protocol, not measured execution duration. The independent model repeat measured9.130321715027094wall seconds and13.640258999999999CPU seconds. Its conservative combined guard/child highwater was6999195648bytes, below8589934592; this process measurement is not an estimate of model memory alone. There are0active ledger rows. Phase deadline1794959581.8433254 remains fixed. No further model runs are authorized in this phase.

The independent reviewer verified unchanged original checkpoint/config/tokenizer hashes. Full embedding/block/final arrays were not retained: boundary restoration is evidenced by matching recorded hashes and zero-difference receipts, while full saved final-token logit vectors were compared bitwise. Working-object restoration is supported by reviewed runner checks, not a separate retained memory snapshot.

This delivery uses existing Python standard-library, PyTorch and pinned tokenizer components. No external repository/tool integration, new package install, training, SAE or general-purpose circuit-discovery system is claimed. Static atlas associations plus one controlled intervention form the bounded investigative workbench; they do not constitute full understanding of the model.

Final A6 exact-document/identity review and controller acceptance passed. A valid independently repeated negative meets this bounded investigation requirement. No extra experiment is proposed.
