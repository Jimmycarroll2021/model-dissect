## CONTINUATION-001 tasks — bounded response-window diagnostic

- [X] T076 [US-C1] Freeze four-case/256-token protocol and two-grader rubric in evidence/continuation/response-window-protocol-v1.md.
- [X] T077 [US-C1] Implement src/minimind_re/continuation_probe.py, charged full-boundary long-prefix parity, and tests/test_continuation_probe.py.
- [X] T078 [US-C1] Stage new continuation guard/policy/packets with three-history locking and tests/test_continuation_guard.py.
- [ ] T079 [US-C1] Independent exact-candidate review and producer admission in evidence/continuation/review/.
- [ ] T080 [US-C1] Execute producer once; save prepared input IDs, parity tensors/checks and all four raw responses in staged runs/.
- [ ] T081 [US-C1] Independent parity/response repeat and two-grader semantic rubric with disagreement AMBIGUOUS.
- [ ] T082 [US-C1] Deliver docs/continuation-result.md and exact final review; send verified status to separate state owner; choose next broad-goal step from evidence.

Dependencies T076→T077/T078→T079→T080→T081→T082. Models serial; no tokenizer/model runtime yet. Broad goal remains active until its own acceptance, not this diagnostic alone.

---

## Current amendment — BEHAVIOR-002 tasks (2026-11-18)

- [X] T068 [US-B2] Freeze completed-response protocol and grammar in evidence/behavior-completion/protocol-v1.md and protocol-v1.json.
- [X] T069 [US-B2] Implement new src/minimind_re/response_probe.py and synthetic tests/test_response_probe.py; preserve earlier computation source.
- [X] T070 [US-B2] Stage isolated response guard/policy/packets and historical chain; validate tests/test_response_guard.py and record initial corrected test-fixture failures.
- [X] T071 [US-B2] Independent exact protocol/runner/guard/fixture/identity review and explicit screen admission in evidence/behavior-completion/review/.
- [X] T072 [US-B2] Execute admitted response-screen-v1; save raw response/step journal and receipt in staged runs/.
- [X] T073 [US-B2] Record qualification NOT_RUN_NO_SELECTED_FAMILY after screen stop; no qualification process or fallback family executed.
- [X] T074 [US-B2] Independent response-independent-v1 repeat and independently recomputed output/identity/budget review.
- [ ] T075 [US-B2] Deliver docs/behavior-completion-result.md, final exact-manifest review and accepted status in TASK_STATE.md.

Order T068→T069/T070→T071→T072→conditionalT073→T074→T075. Runtime has no parallel stages. Negative screen skips T073 model work but still requires T074. All previous tasks/outcomes below remain historical evidence; no old slot is reused.

---

# Active task amendment — ATLAS-001

T059–T063 completed and independently accepted through evidence/atlas/atlas-acceptance-v1.json. Paths below describe intended deliverables; final receipt links the actual staged output/review paths. T064–T066 independently complete through evidence/atlas/review-a5-independent-result-v1.json: scientific REFUTED, procedural PASS. T067 is checked following independent A6 delivery PASS and controller acceptance in evidence/atlas/final-review-v1.json and evidence/atlas/controller-acceptance.json.

- [x] T058 [US1] Reconcile prior outcomes and write acceptance/budget contract in evidence/atlas/protocol-v1.md; planning independently accepted; A1–A4 runtime acceptance in evidence/atlas/atlas-acceptance-v1.json.
- [x] T059 [P] [US1] Implement atlas CLI and synthetic tests for full tensor accounting, bounded statistics/spectral/vocabulary/QKOV analyses; freeze candidate manifest under evidence/atlas/.
- [x] T060 [P] Reconcile immutable historical runtime ledger and implement separately admitted ATLAS-001 guard/ledger; save executable migration and failure checks under evidence/atlas/.
- [x] T061 Independently review exact atlas/guard/protocol candidates and reserve phase identity/clock before first actual checkpoint analysis; save evidence/atlas/admission-review.json.
- [x] T062 [US1] Run admitted atlas and persist JSON/readable report plus raw selection and identity receipts under runs/atlas/ and docs/.
- [x] T063 [US1] Independently verify atlas complete91-key coverage, numerical samples, dimensions/semantics, bounds and errors; save evidence/atlas/atlas-review.json.
- [x] T064 [US3] Select one atlas-informed investigation and freeze detailed protocol/selection algorithm/metrics/controls plus evaluator-owned fresh fixtures under evidence/atlas/ before any new forward.
- [x] T065 [US3] Independently review exact investigation and resource packets, then execute baseline/no-op/target/norm-matched/restored development and fresh confirmation runs under runs/atlas/.
- [x] T066 [US3] Independently repeat frozen candidate, recompute supported/refuted/inconclusive verdict, and audit restoration/identities/lifetime totals under verification/.
- [x] T067 Deliver atlas and investigation evidence with tested quickstart and independent exact final document review; coordinator records A1–A6 and native goal status in TASK_STATE.md.

Dependencies:T058→(T059 || T060)→T061→T062→T063→T064→T065→T066→T067. Distinct owners only; no nested agents or concurrent model runs. T064 detailed freeze may split development selection from intervention freeze only with predetermined selection rules. No historical failed checkbox changes. A bounded negative A5 outcome can complete T066; failure to execute required controls/fresh confirmation cannot.

# Current task amendment — BEHAVIOR-001

- [x] T050 [US3] Independently review frozen protocol evidence/behavior/protocol-v1.md and exact budget before fixtures/runs.
- [x] T051 [US3] Freeze all fresh screening/qualification/development/intervention fixtures and tokenizer-format evidence under evidence/behavior/; evaluator owns holdouts.
- [x] T052 [US3] Implement minimal runner adaptation and synthetic checks, preserving forward/hook identity; save exact review under evidence/behavior/.
- [x] T053 [US3] Run fixed screen and preserve STOP_NO_QUALIFIED_PAIR in runs/behavior/screen-v1.json; fresh qualification NOT_RUN under the frozen conditional rule.
- [ ] T054 [US3] NOT_RUN / inapplicable after T053 stop: development observation and coordinate/control selection under evidence/behavior/.
- [ ] T055 [US3] NOT_RUN / inapplicable after T053 stop: development and heldout five-arm interventions in runs/behavior/.
- [x] T056 [US3] Independently repeat screening and compute STOP_NO_QUALIFIED_PAIR; exact48 rows PASS in evidence/behavior/independent-screen-review.json.
- [x] T057 [US3] Report delivered in docs/behavior-qualification-result.md and TASK_STATE.md; exact-document review PASS in evidence/behavior/final-document-review.json and conditional controller acceptance recorded. Bounded phase complete at valid stop; larger understanding objective not achieved.

Dependencies T050→T051; T052 may parallel T051 with separate ownership, then both→T053→T054→T055→T056→T057. A T053 negative goes directly to applicable T056 and T057; unused dependent tasks NOT_RUN. No authorisation from historical closed slots. Planning leaf owns specification only; source/fixtures/runtime assigned separately.

# Current task amendment — MECHANISM-001

Historical tasks below retain their original outcomes; numerical-repair/R2 blockers do not govern this explicitly approved narrower slice. No old checkbox is marked successful by scope revision. Follow evidence/mechanism/protocol-v1.md; original R2 FAIL remains.

## US2 — same-mode fidelity and actual tool

- [x] T040 [US2] Independently review amendment and frozen protocol in evidence/mechanism/planning-review.json.
- [x] T041 [US2] Implement named native PyTorch hook seam and capture/intervention lifecycle in src/minimind_re/forward.py and mechanism runner; freeze candidate manifest in evidence/mechanism/implementation-manifest.json.
- [x] T042 [US2] Verify synthetic capture, single-position isolation, cleanup-after-exception and bitwise removal in tests; save exact results under evidence/mechanism/.
- [x] T043 [US2] Evaluator freeze fresh M2 and16development/32heldout fixtures with expected/foil token labels under fixtures/mechanism/; record identities and separation in evidence/mechanism/fixture-freeze.json.
- [x] T044 [US2] Independently review exact candidate, guard bounds and preregistration before guarded same-mode M2/tool run; save all comparisons in runs/mechanism/m2.json.

## US3 — one hypothesis, all controls and delivery

- [x] T045 [US3] Run one development baseline observation and freeze deterministic target/control selection and statistics in analysis/mechanism-hypothesis.json.
- [x] T046 [US3] Run fixed five-arm development pass without reselection; save raw results in runs/mechanism/development.json.
- [x] T047 [US3] Evaluator run all32heldout inputs/five arms and frozen metrics in runs/mechanism/heldout.json.
- [x] T048 [US3] Separate verifier repeat exact M2+heldout and independently recompute verdict, provenance, norms, restoration and limits in verification/mechanism-final.json.
- [x] T049 [US3] Deliver tested coordinate explanation, arm table, supported/refuted/inconclusive result and unresolved cache/full limitation in docs/mechanism-investigation-result.md; coordinator records amended acceptance in TASK_STATE.md.

Dependencies: T040→T041→T042; T043 can proceed independently after T040 with separate fixture owner, no holdout exposure. T042+T043→T044→T045→T046→T047→T048→T049. Failed prerequisites stop dependent runtime and still require truthful partial delivery. Only model-free synthetic tests may precede T044. No new training/download/deadline. Owner of runtime ledger must serialize all admitted execution; no model parallelism.

T040–T047 completion is supported by evidence/mechanism/planning-review.json,implementation-review-v1.json,fixture-freeze.json,m2-result-review.json,development-selection-review.json,development-arms-review.json,producer-result-review.json. T048 independent runtime repeat completed: evidence/mechanism/independent/execution-result.json,40postrun identities matched,allpairedrows exact,scientific INCONCLUSIVE. T049 final reviewed delivery complete: docs/mechanism-investigation-result.md and evidence/mechanism/final-document-review-final.json; bounded runtime evidence previously accepted by coordinator. No further model runs or hypotheses. M2 PASS is narrower than original R2; heldout result INCONCLUSIVE establishes no mechanism.

## Historical tasks preserved

# Tasks: Tiny model reverse engineering

Input: plan.md, spec.md, research.md, data-model.md and contracts/local-workflow.md.
Status: implementation and evaluation partial; planning I1 correction independently reviewed separately. Checkboxes below reflect established deliverables, not full R0–R4 acceptance. R2 FAIL; R3 NOT_RUN. Independent numerical verdict recorded; functional CLI delivered with synthetic evidence.

## Phase 1: Setup

- [ ] T001 Reconcile readiness and exact ownership/resource reservations in TASK_STATE.md using evidence/execution/readiness-report.md; retain existing caps and historical receipts.
- [x] T002 Record the compatible existing Python/dependency environment and import smoke checks in runtime/environment.json; no global installs or duplicated caches.

## Phase 2: Foundational gates

- [x] T003 Implement cumulative preflight/reservation/accounting/deadline/owned-process stop behavior in runtime/guard.py; record all candidate/oracle calls, arms, repeats, token/memory/disk/download spend.
- [x] T004 Exercise cap exhaustion, interrupted receipt recovery and foreign-process preservation with harmless fixtures in runtime/test_guard.py; save evidence in runtime/guard-checks.json before model forwards.
- [x] T005 Acquire only selected compatible checkpoint plus pinned source/config/tokenizer into artifacts/original/; verify expected download hashes and create artifacts/manifest.json; oracle owner inspects source privately and confirms compatibility (R0).
- [x] T006 Freeze allowed evidence and prior exposure in analysis/exposure-ledger.json, then publish only declared views under artifacts/allowed/; never give investigator source/module tree or withheld architecture fields (R0).
- [ ] T007 Implement oracle/adapter.py and oracle/protocol.json for deterministic FP32 interface, masked positions, cache, named internal observations and query receipts; oracle owner checks interface without exposing source.
- [ ] T008 Freeze fixture identities and observation-budget rules in fixtures/protocol.json: 12 parity cases, 16 development inputs, 32 private held-out inputs with counterexamples; guard every forward and defer hypothesis fields until development evidence.

## Phase 3: US1 — evidence-backed map (P1)

Independent test: exact stored tensor key coverage, claim provenance, oracle correct/incorrect/unresolved ledger; unresolved required computation blocks R1 completion.

- [x] T009 [US1] Add inventory accounting/missing-tensor/alias negative checks in tests/test_inventory.py using harmless synthetic tensors; no target forward required.
- [x] T010 [US1] Implement native safe checkpoint inspection in src/minimind_re/inventory.py and save all tensor records to analysis/inventory.json (R0–R1).
- [x] T011 [US1] Build analysis/structural-map.json and analysis/structural-map.md with operation/edge evidence, alternatives, uncertainty and all tensor mappings; freeze hash before oracle structural comparison (R1).
- [x] T012 [US1] Oracle owner compares the frozen map against pinned source and records correct/incorrect/unresolved in runs/evaluation/map-review.json; disclosed feedback invalidates affected blind-inference claims (R1).

## Phase 4: US2 — reconstructed computation (P2)

Independent test: all real-token logits and embedding/block/final captures on 12 fixtures, cached/full-sequence continuation, elementwise tolerances and exact top-1; no-op/restored match; source delegation absent.

- [ ] T013 [US2] Encode mask/cache/position and nonfinite comparison contract checks in tests/test_contract.py; ensure missing/unsupported fixture paths fail instead of skipping.
- [x] T014 [US2] Implement independent primitive tensor composition and cache/capture interface in src/minimind_re/forward.py, derived from frozen map; never import oracle/target forward (R2).
- [x] T015 [P] [US2] Oracle owner implements paired numerical comparator in oracle/evaluate.py, preserving every required row, max/mean errors, mismatch counts and failure receipts (R2).
- [ ] T016 [US2] Freeze candidate source/map/environment manifest in analysis/candidate-manifest.json; evaluator runs guarded R2 baseline/no-op/restored suite to runs/evaluation/parity.json, preserving failures and originals (R2).

## Phase 5: US3 — one internal hypothesis (P3)

Independent test: frozen directional quantitative rule, baseline/no-op/target/matched internal control/restoration, paired development and held-out results, specificity against broad damage, supported/refuted/inconclusive verdict.

- [ ] T017 [US3] Collect only preregistered development observations under the common guard into runs/development/observations.json; freeze one site/hypothesis/metric/effect floor/control matching/specificity/decision rule in analysis/probe-protocol.json before probe-arm or held-out execution (R3).
- [ ] T018 [US3] Implement reversible activation perturbation and guaranteed hook removal/restoration in src/minimind_re/probe.py; exercise no-op, exception restoration and invalid site rejection in tests/test_probe.py (R3).
- [ ] T019 [US3] Run frozen paired arms on development then evaluator-owned held-out inputs into runs/development/probe.json and runs/evaluation/probe.json; save raw paired data, counterexamples and the preregistered verdict without threshold tuning (R3).

## Phase 6: independent completion and handoff

- [ ] T020 Freeze exact final artifact/source/protocol/run manifest in verification/candidate-manifest.json; independent verifier audits leakage, untouched originals and cumulative budgets, repeats required R2/R3 within reserved allocation, and saves verification/checks.json plus verification/report.md (R0–R4).
- [ ] T021 Validate actual runnable commands against specs/001-minimind-reproduction/quickstart.md, deliver map/candidate/raw receipts/limitations, and let coordinator record accepted/partial/blocked in TASK_STATE.md from independent evidence only (R4).

## Dependencies and parallel execution

T001→T002→T003→T004; acquisition T005 can proceed after readiness without model forward, then T006. T007/T008 require identities and tested runtime guard before their model observations. T009/T010 need allowed artifacts; T011→T012 completes US1. T013→T014 and T015 can proceed in parallel after interface/map freeze with exclusive candidate/oracle ownership; T016 waits for both. T017 depends on parity and development observation protocol; T018→T019 follows hypothesis freeze. T020→T021 completes delivery. No concurrent GPU execution: runtime owner serializes reservations.

US1 example: investigator writes map while oracle prepares comparison tooling without reading unfinished claims; actual comparison waits for freeze. US2: T014 candidate and T015 comparator parallel in disjoint directories. US3: after protocol freeze, investigator builds perturbation while evaluator prepares private matched fixtures; neither runs held-out until candidate freeze. These are parent-dispatched leaves, never nested delegation.

## Strategy and task accounting

21 tasks: setup 2, foundation 6, US1 4, US2 4, US3 3, completion 2. US1 is the first useful partial milestone; all three stories and R4 are the requested outcome. Scope remains one candidate/hypothesis plus at most one diagnosed retry; do not create a candidate search. Changes from oracle feedback are source-assisted, preserve failures, require fresh held-out evaluation within caps. Missing identity/capacity gates stop dependent work but permit source-free local implementation. A refuted/inconclusive hypothesis is acceptable investigation, not a positive mechanism discovery.

## Execution evidence checkpoint

Checked work is supported by `evidence/execution/acquisition.json`, `implementation-milestone1.json`, `implementation-validation-fix.json`, `runtime/environment.json`, `analysis/exposure-ledger.json`, `runs/evaluation/map-review.json` and `oracle-parity-summary.json`. T003/T004 are supported by independent guard tests and repaired guard review; the guard remains polling-based and interrupted ledgers fail closed. T007/T008 remain open because interface validation and full probe protocol admission are not complete. T016 has executed but remains unchecked because required R2 comparisons FAIL. T017 and T019 are blocked/NOT_RUN; T018 has a generic five-arm runner and synthetic restoration tests but stays unchecked because target-specific protocol/control validation is absent; T020 has an independent partial verdict but stays unchecked because the full R0–R4 outcome is not accepted; T021 has a runnable guide and functional CLI receipts; remains unchecked pending independent final-document review and coordinator partial handoff. Unchecked setup/reconciliation work requires coordinator confirmation rather than inferred completion.

T013 partial coverage: shape, dtype, context and all-masked-input checks are implemented/tested. Explicit nonfinite and cache contract test coverage is missing, so T013 remains unchecked; measured parity does not substitute for these missing negative tests.

## Continuation: US2 discrepancy, then US3 delivery

- [x] T024 Reconcile original elapsed window, ledger/process/lock state and preserve setup receipts in evidence/continuation/reconciliation.json and speckit-setup.json.
- [x] T025 [US2] Preregister distinct diagnostic hypotheses, unchanged acceptance and promotion/stop rules in evidence/continuation/numerics-protocol-v1.json.
- [x] T026 [US2] Independent reviewer binds exact protocol identity in evidence/continuation/protocol-review-v1.json; execution admission retains shared runtime guard.
- [x] T027 [US2] Complete original three guarded diagnostics D1–D3; results and reviews retained in evidence/continuation/diagnostics/ and runs/continuation/.
- [ ] T028 [US2] Promote at most one evidence-supported source/environment candidate, freeze exact identity and fresh evaluator parity fixtures, then run complete unchanged R2 to runs/evaluation/parity-continuation.json.
- [ ] T029 [US2] Separate verifier repeats exact candidate R2 and preserves verdict in verification/parity-continuation.json.
- [ ] T030 [US3] On R2 PASS only, complete T017–T021 using preregistered development/hypothesis/held-out controls; save analysis and evaluation receipts.
- [ ] T031 Complete original delivery tasks and independent R0–R4 bundle verdict, update TASK_STATE.md and quickstart with established outcomes.

Dependencies: T024–T029 sequential; no model diagnostic before T026. T030 requires T029 PASS. T031 reports failures truthfully if required gates block. Existing passed unchanged evidence may be reused; checkboxes never imply runtime acceptance.

## Protocol-v2 evidence-driven continuation

T028 executed strict full R2 but remains unchecked because234rows failed; T029 was not run for that rejected candidate. T030 remains blocked. Original protocol and failures are immutable.

- [x] T032 [US2] Independently review exactly one additional D4 script and finite amendment in evidence/continuation/protocol-v2-review.json.
- [ ] T033 [US2] Interpret and independently review completed D4 receipt runs/continuation/D4-attention.json; completion alone cannot establish the attention-only cause.
- [ ] T034 [US2] If D4 supports the cause, freeze one separately implemented candidate/reference attention policy and fresh parity fixtures; preserve native-prefill equivalence and original padded-fixture R2 coverage; report cross-padding invariance separately as a diagnostic (verification/runtime-static-review-v1.json), correcting the broader proposal wording.
- [ ] T035 [US2] Run full unchanged native-plus-R2 acceptance and separate exact-candidate repeat; preserve every failed check and original lifetime resource totals.

T032→T033→T034→T035→T030. D4 alone adds4096reservations/300seconds; total4diagnostics/16384reservations. The one conditional policy candidate allows at most2full runs of18000reservations/600seconds each; any needed additional native-check reservation must be exact and declared before admission. Original deadline1794912024.8448846 and lifetime1,000,000tokens remain fixed. No automatic second policy candidate; no mechanism execution before independently passing complete R2.

## Latest attempt accounting and remaining acceptance

T034 canonical candidate/reference implementation and fixture freeze were attempted, but its native-prefill gate failed6/16; do not mark T035 complete. Exact v2 restoration passed6manifest entries and7synthetic tests. CPU strict234failure and GPU-native30failure receipts remain authoritative. T033 awaits the separate final evidence audit; diagnostic completion alone is insufficient. T028/T029/T030/T035 and original R2/R3 completion remain unchecked. No development observations or mechanism hypothesis exist.

- [x] T036 [US2] Preserve canonical rejection and exact v2 restoration in evidence/continuation/candidate-fix-v3-restoration.json; archive rejected v3 without deleting assertions.
- [x] T037 [US2] Complete independently reviewed native-only GPU feasibility and diagnosed UUID repair attempt in evidence/continuation/gpu-native-protocol-v2-outcome.json; preserve failed initialization and all charges. This is an executed failed diagnostic, not R2 PASS.
- [x] T038 Document attempted remedies, current identity, bounded resource totals and unresolved gates in docs/numerics-continuation-result.md.
- [x] T039 Independent acceptance audit completed in evidence/continuation/acceptance-audit.md: retain R2 FAIL; comparator not proven invalid; explicit material scope decision required. Audit completion is not acceptance success.

Next dependency: explicit user scope decision following completed T039, then only an independently reviewed bounded protocol consistent with that answer. No pending task authorizes relaxed thresholds or unrecorded experiment retries. Full Spec Kit delivery remains partial.

## ECHO-MECHANISM-001

- [x] T083 Freeze24fresh matched digit pairs, scientific scoring/control rule and exact full-prefix budget in evidence/echo-mechanism/protocol-v1.md.
- [x] T084 Implement native attention-output slice hooks, charged observation/intervention runner and additive four-history guard; synthetic checks only.
- [ ] T085 Independent exact-candidate protocol/source/resource review and producer admission.
- [ ] T086 Execute producer exactly once under frozen guard, retaining all conditional stops and incremental outputs.
- [ ] T087 Independent exact repetition of every full-vocabulary vector, selection, held outcomes and history/resource reconciliation.
- [ ] T088 Deliver bounded causal result and truthful mechanism limits; independent final artifact review.

## ECHO recovery after failed producer

T086 attempted once and stopped RESOURCE_DISK_CAP; T087/T088 did not execute for the failed phase. Original reservations/evidence retained; no original scientific acceptance claimed.

- [x] T089 Reproduce backing-storage serialization defect synthetically and preserve versioned owning-clone fix, exactdiff and storage regression receipts.
- [x] T090 Stage clean ECHO-RECOVERY-001 under explicitly approved1,038,544 ceiling, same24pairs/scientificprotocol, fiveimmutablehistoricalchains,76800perrole; syntheticguardchecks.
- [ ] T091 Independent exact recovery-candidate admission.
- [ ] T092 Fresh full producer once under recoveryguard; preserve conditionalstops and everyvector.
- [ ] T093 Independent full repeat and resource/history reconciliation; original scientific gate unchanged.
- [ ] T094 Deliver accepted bounded result/limitations and final exactartifactreview.

## Cache operand diagnosis

- [x] T095 Stage scientific cacheoperand source with liveledger/PID/deadline binding, six-history guard, strictMKL environment and exactpackets; source/guardsyntheticchecks only.
- [ ] T096 Independent exact cachecandidate review/admission.
- [ ] T097 Run producer diagnostic once, preserving rawoperands/accounting.
- [ ] T098 Independent numerical trace repeat and failure-localization review.
- [ ] T099 Deliver diagnosed/not-fixed result and exact historicalR2 limitation.

## Shape-matched cached-attention limited experiment

- [x] T100 Implement versioned candidatecachedattention-only querywidth policy, unchangedfullbranchASTproof and syntheticchecks; integrate independentlyowned referencepolicycontext.
- [x] T101 Stage fixed12call/63position probe,128perrole reservation, seventh-history guard and exactprotocol/packets; no modelcalls.
- [ ] T102 Independent exactcandidate admission.
- [ ] T103 Producer limitedfixture execution underguard, no retry or numericaltolerancerelaxation.
- [ ] T104 Independent all-arrayrepeat and fullscopeaudit.
- [ ] T105 Deliver limitedmode outcome without upgrading historicalnativeR2FAIL or claiming fullmatrixcoverage.

## Proposed complete matrix validation

- [x] T106 Prepare bothoriginal12fixture matrices, all3realhookarms, untouchednativefidelity, versionedsource and independentreference integration; freeze50526/1656/9120scope andfullraw/losslessKVstorage.
- [x] T107 Stage proposed-only8historyguard/policy/packets withuser_budget_approved=false, prospective1137888ceiling/4GiBphase; runsyntheticchecks.
- [x] T108 Independent exact static review PASS in review/static-approval-review-v1.json; user resource approval recorded below. T109 runtime admission/execution remains incomplete.
- [ ] T109 Activateonlyapprovedpolicy, finalexactadmission, producerandindependentfullmatrixruns.
- [ ] T110 Independentrawarrays/cachechunkreconstruction andgateaudit; delivertruthfulnewmodevalidation/nativeR2boundary.


## User-approved complete matrix resource amendment (2026-11-18)

User instruction **“Scale and big as you need”** answered the concrete pending request. Apply only the reviewed CACHE-SHAPE-MATRIX-001 lifetime ceiling **1,137,888** and **4 GiB** phase disk cap. Preserve all **1,036,688** historical charged positions including failed runs; reserve **50,600** each for producer and independent repeat (**101,200** combined), reaching the ceiling exactly. The frozen full matrix remains **24 fixtures × 3 arms, 50,526 actual positions / 1,656 calls / 9,120 rows per role**. CPU two threads, 8 GiB combined owned RSS, 8 GiB combined disk, 2 GiB free reserve, 900 seconds per role, 1,800 aggregate model seconds and 3,600 seconds phase elapsed remain unchanged. No scientific protocol, numerical tolerance, code, historical ledger, or old failure is changed.

Approval-only policy and versioned packet rebinding is prepared in `evidence/cache-shape-matrix/candidate-manifest-approved-v1.json`; proposed packets and policy snapshot are preserved. Independent exact runtime admission is still required before execution; no phase ledger/clock or model run is started by this amendment. Prior pending-approval paragraphs are historical. Original native R2 FAIL remains separate from any future result for this numerical mode.


## CACHE-PREFIX-WIDTH-001 original R2 continuation

User instruction “Keep going until this job is done” following the repair-versus-limited-acceptance choice is applied as continuation of original numerical parity repair, not waiver. Preserve original R2 and full new-mode matrix FAIL plus all1,138,144historical reservations. The next bounded discriminator compares the exposed failing left-padded length-one fixture's full passes at physical widths6and8, targetslot5, unchanged candidate/native arithmetic. Both actors total28actualpositions/4calls perrole,64reserved each/128combined, controller-approved ceiling1,138,272. Frozen protocol `evidence/cache-prefix-width/protocol-v1.md`; no model run before independent exact admission. Newphase1800seconds,300seconds perrole,CPU2,ownedRSS8GiB,phase512MiB,combineddisk8GiB,free2GiB; no GPU/download/training/retry or threshold relaxation. Ten immutable historical chains are preserved. Stop after the discriminator/repeat and report earliest unequal operation with direct-input equality; no automatic repair or arbitrary acceptance change. This amendment does not mark old numerical gates passed.

- [x] T111 Prepare unchanged-fullwidth layer0 observation candidate, frozenfixture/protocol and ten-history additiveguard; syntheticchecks only.
- [ ] T112 Independent exact runtime admission of CACHE-PREFIX-WIDTH-001.
- [ ] T113 Execute28position producer once, preserving naturaloperation traces and originalbaselinefidelity.
- [ ] T114 Independent exactrepeat/rawanalysis; qualify earliestdifferentoperation by directinputequality and preserve originalR2FAIL until fullrepair acceptance.

## Current versioned repair delivery — superseding execution checklist

Earlier unchecked attempt-specific acceptance tasks remain historical where their candidates failed or were superseded; they are not silently marked successful by a different candidate. This list owns the current softmax-mode delivery status; immutable receipts own each prior outcome.

- [x] T115 Preserve prefix observation failures/charges; complete independently repeated natural-operation localization and model-free fixed-policy replay.
- [x] T116 Implement frozen cached-softmax candidate and separately authored reference; pass synthetic/source-branch checks and independent exact admission.
- [x] T117 Complete two-fixture28call88position screen and independent repeat:80comparison rows and560saved tensors exact.
- [x] T118 Freeze unchanged24fixture/3arm/9120row full matrix,14historical chains and explicit resource amendment; obtain independent exact admission.
- [x] T119 Execute producer once:1656calls50526actualpositions,9120PASS/0FAIL; retain allraw/compactKV evidence.
- [x] T120 Complete independent full repeat, exact raw reconstruction/comparison and resource reconciliation; receipt evidence/cache-softmax-matrix/review/independent-repeat-final-v1.json confirms 18,144 exact tensors, all 9,120 rows PASS and 1,239,856 charged.
- [x] T121 Freeze integrated repair report/workbench/README/spec-plan handoff and independent whole-objective R0–R4/B1–B6 audit. Controller accepted PASS_WHOLE_OBJECTIVE_WITH_DECLARED_SCOPE in evidence/goal-coverage/whole-objective-review-v2.json; native goal lifecycle closure and closure-overlay verification remain separate.

Original native R2FAIL remains; current mode acceptance, when independently confirmed, applies the original numerical gates to the actual repaired execution mode without threshold or coverage waiver.
