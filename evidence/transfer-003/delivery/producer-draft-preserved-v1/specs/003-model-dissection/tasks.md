# Tasks: third model dissection

Initial dependency plan; GPT-2 selection reviewed, acquisition assigned separately, target runtime not admitted.

## Phase 1: setup and research

- [x] T001 Preserve old coordination bytes in evidence/transfer-003/state/preservation-manifest-v1.json and prepare preliminary specs/003-model-dissection/spec.md.
- [x] T002 Capture read-only host inventory in evidence/transfer-003/state/host-readiness-v1.json.
- [x] T003 Integrate primary repository/paper research and freeze model selection in specs/003-model-dissection/research.md.
- [ ] T004 Independently review selection, acquisition bounds and preliminary scope in evidence/transfer-003/review/selection-v1.json.

## Phase 2: foundations

- [x] T005 Acquire admitted pinned artifacts in a dedicated EXTERNAL_ARTIFACT_ROOT workspace and independently audit identity in evidence/transfer-003/review/acquisition-v1.json.
- [ ] T006 Freeze small calibration contract, measured resource gates and real lossless serialization roundtrip in specs/003-model-dissection/runtime-protocol.md; independently review exact runtime build before execution.

## Phase 3: US1 weight and computation atlas

- [x] T007 [US1] Produce exhaustive tensors, ties and computation provenance in evidence/transfer-003/atlas/manifest-v1.json.
- [x] T008 [US1] Independently audit tensor coverage and graph assertions in evidence/transfer-003/review/atlas-v1.json.

## Phase 4: US2 faithful editable computation

- [ ] T009 [US2] Build independently composed candidate and separately owned native reference, binding identities in evidence/transfer-003/build/manifest-v1.json.
- [x] T010 [US2] Run admitted small calibration including singleton/long full-prefill and actual serialization; record timing/memory/bytes/guard overhead in evidence/transfer-003/calibration/result-v1.json.
- [x] T011 [US2] Freeze measured core numerical matrix and resource caps in specs/003-model-dissection/numerical-protocol.md, then obtain independent admission.
- [ ] T012 [US2] Execute native-faithful core matrix and actual walkthrough in evidence/transfer-003/numerical/producer-v1.json and evidence/transfer-003/trace/manifest-v1.json.
- [x] T013 [US2] Independently repeat exact candidate and recompute retained raw comparisons in evidence/transfer-003/review/numerical-v1.json.

## Phase 5: US3 controlled investigation

- [ ] T014 [US3] Qualify discovery fixtures and freeze hypothesis, controls, eligibility, metrics, 16 held-out cases or justified alternative in specs/003-model-dissection/internal-protocol.md.
- [ ] T015 [US3] Execute admitted controlled intervention, no-op/restoration and identical candidate/native edits with bounded intervention-equivalence check (any native-only limitation requires explicit disclosure) in evidence/transfer-003/science/producer-v1.json.
- [ ] T016 [US3] Independently repeat scientific experiment and recompute qualified held-out result in evidence/transfer-003/review/science-v1.json.

## Phase 6: US4 delivery and stopping

- [ ] T017 [US4] Produce bounded report and tool reuse matrix in specs/003-model-dissection/result.md.
- [ ] T018 [US4] Bind exact deliverable manifest and independently audit S0–S5 in evidence/transfer-003/delivery/final-review-v1.json; close TASK_STATE.md only after acceptance.

## Dependencies and parallel opportunities

T001–T004 precede acquisition; T005–T006 precede target runs. Static atlas and separately owned candidate/reference builds may proceed in parallel after artifact admission. T010 calibrates before T011 large-grid freeze. T012–T013 precede held-out scientific execution, while discovery protocol drafting can proceed independently. T016 precedes final audit. Research workers own research paths; state writer owns shared coordination files. Optional cached backend work requires a separate protocol and cannot silently become a core blocker. No new agent is launched by this leaf.

## Independent test criteria

US1: exhaustive tensor identity and evidence-labelled graph. US2: every core comparison passes with actual independent execution and replayable trace. US3: qualified held-out evidence, controls and independent repeat; all-ineligible is incomplete. US4: exact final-manifest audit and scoped report. MVP learning increment is atlas plus small calibrated faithful execution, not final completion.

## Full numerical checkpoint

Independent full-review-v2.json SHA256fa9c83e11fd60e2b94c5350f464c05c0337c5cda9ccc697d0edc83e65d874d62 completes T013 and the numerical portion of T012; T012 stays unchecked because actual-input walkthrough is still pending. T010/T011 acceptance is supported by the completed admitted full matrix and preserved pilot receipts. T005 acquisition audit previously passed. Target atlas T007/T008, trace remainder, scientific T014–T016 and final delivery remain pending. No science/static/tokenization target result is inferred from numerical PASS.

## Atlas acceptance and active science checkpoint

T007/T008 accepted under atlas-actual-v2.json SHA25667a97c9740b7d1ca29f1275029d21565ba1e20b1eac900f5c38dbbd3bf99cafc with explicit derived population-std supplement caveat. Token fixtures admitted and independently decoded, all24pairs8tokens per prompt. Scientific producer active per coordinator; T014–T016 remain unchecked until eligibility/selection/execution/repeat receipts establish their requirements. Trace and final delivery remain pending. No positive/negative target scientific finding yet.
