# Data model

All records include schema_version, candidate_id, created_utc and provenance. Hashes are full SHA-256; source revisions full Git identifiers. Unknown required identity blocks runs.

| Entity | Required fields / validation | Relationships |
|---|---|---|
| ArtifactManifest | role, path, bytes, sha256, origin, license/terms pointer, immutable flag; checkpoint/config/tokenizer/source/environment completeness | Parent of inventory/run identities |
| EvidenceLedger | allowed inputs, withheld paths, prior knowledge, source/config exposure, feedback timestamps and claim IDs | Every map claim cites ledger observations |
| TensorRecord | exact name, shape, dtype, elements, storage bytes, alias group, mapped role or unresolved reason | Exactly covers checkpoint tensor key set |
| MapClaim | operation/edge, evidence IDs, status, competing explanations, required-path flag | Required path cannot retain unresolved links at R1 completion |
| Protocol | fixture hashes, masks/position/cache semantics, sites, dtype/backend, tolerances, hypothesis/direction/threshold/control/specificity rules, version and freeze timestamp | Frozen before relevant observations |
| RunReceipt | candidate/artifact/protocol hashes, argv, environment, arms, start/end, token count, GPU/CPU/wall time, peak memory, disk/download ledger, exit/stop reason | Persist before reporting; include failures |
| Comparison | fixture/arm/site IDs, finite status, max/mean absolute/relative error, elementwise failures, top1 mismatch count | R2 PASS only if all required rows pass |
| ProbeResult | paired baseline/noop/target/control/restored metrics, counterexample group, aggregate effects, uncertainty and verdict | R3 supported/refuted/inconclusive under frozen rule |
| Verification | exact manifest, requirement IDs, PASS/FAIL/NOT_RUN, commands, evidence paths, limitations | Independent verifier only |

Transitions: planned → ready (R0/protocol/resource gate) → running → completed/failed/stopped; candidate frozen → evaluated → independently verified → controller accepted/partial. Never overwrite failed receipts. A changed candidate/protocol creates a new version and explicit invalidation; no retrospective threshold changes.


## MECHANISM-001 amendment

MechanismProtocol: id/version/mode, immutable code+model+fixture hashes, slot caps, fixed selection/control/decision rules. Fixture: split/category/inputIDs/mask/positions/expectedID/foilID. SelectedHypothesis: site/residualCoordinate/controlCoordinate/RMS/scale/correlations/freezeTime. ArmRow: fixture+arm identities, margins, expectedNLL, top1, removedNorm, logits summary. Verdict: prerequisite gates plus supported/refuted/inconclusive and independent repeat identity. Transitions: protocol-reviewed→tool/M2-passed→dev-selection-frozen→arms-measured→independently-verified; any gate failure stops dependent execution.
