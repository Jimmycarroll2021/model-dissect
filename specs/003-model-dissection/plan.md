# Implementation plan: third model dissection

Status: PRELIMINARY; no acquisition, install or target runtime admitted by this document.
Model: openai-community/gpt2 at revision 607a30d783dfa663caf39e06633721c8d4cfcd7e, selection independently reviewed PASS. Spec: [spec.md](spec.md).

## Summary

Apply the tested dissection workflow to a somewhat larger third open-source model. Core acceptance is native-faithful full-prefill reconstruction, exhaustive atlas/map, actual trace and qualified controlled intervention with independent repeat. Optional cache/backend work has its own gate. Source-assisted result only; no universal-recovery claim.

## Technical context

Reuse existing Python/PyTorch inference tooling and Spec Kit records where compatible, after dependency inventory. GPT-2 small: advertised 124M parameters, 12 layers, hidden width 768, 12 heads, learned positions, Conv1D projection layout, tied output, vocabulary 50,257 and context 1,024. Advertised parameter count is not an audited stored-tensor count; classify learned parameters, ties and possible causal buffers separately. Source licence reported MIT by selection research; preserve exact model/source licence provenance. Native oracle: installed Transformers GPT2LMHeadModel eager CPU FP32 implementation, exact package/source identity to freeze during model-free build. Candidate architecture adapter and target-specific scientific fixtures follow research. Do not install/download until selection and bounded acquisition review. Working model artifacts and bulk raw arrays should use a new isolated EXTERNAL_ARTIFACT_ROOT workspace after capacity admission; root filesystem stores only small records. No new GSD controller or native goal.

## Constitution check

Constitution 2.1.0 requires identity, preregistration, preservation, independent evaluation and one workflow. Design respects these principles; full planning pass remains BLOCKED on research, resource protocol and independent review. Known-source assistance must be disclosed; separate oracle ownership and independently composed candidate remain mandatory. Earlier Pythia cached/full failure remains immutable. Skill phase-0 delegation is owned by parent coordinator; this leaf does not spawn agents. Detailed initial protocols below now permit model-free synthetic build only; acquisition and runtime retain separate independent gates.

## Resources and admission

Read-only snapshot: evidence/transfer-003/state/host-readiness-v1.json. Initial observation: EXTERNAL_ARTIFACT_ROOT about 50.6 GB available; system filesystem about 1.76 GB available; RAM available about 14.4 GB; swap almost full; RTX 4090 24,564 MiB total, 1,223 MiB used and 21% utilization. Active browser/desktop/other processes must remain undisturbed. These are transient observations, not reservations or safe runtime clearance. No GPU compute-process rows were returned; this does not imply exclusive GPU ownership.

Before execution freeze CPU/GPU choice, threads, owned memory, minimum free reserve, positions/calls, acquisition/disk ceilings, per-packet and aggregate seconds, stop conditions and verification allowance. Defaults are not borrowed uncritically from Pythia. Require refreshed readiness and exact independent build review. Insufficient resources stop dependent work; no cleanup or process termination authority is inferred.

## Phase 0 — research and frozen scope

Integrate separately owned evidence/transfer-003/research findings. Select one bounded target, freeze identifiers and native reference, document paper/repository claims versus local evidence, then independently review selection and acquisition packet. Preserve previous exact state in state/preservation-manifest-v1.json. Selection evidence is recorded in research.md; independent selection review PASS; acquisition audit remains pending.

## Phase 1 — static identity and small executable proof

Acquire only admitted artifacts; audit hashes/licence/format. Build atlas and map; construct independent editable candidate and separately owned reference harness. First run a small calibration of representative singleton and longer full-prefill inputs. Test actual raw serialization/reload and reviewer recomputation before allocating large grids. Measure wall time, peak memory, raw bytes and guard overhead. Freeze full matrix and caps from measurements, never from successful command status alone.

## Phase 2 — core execution and science

Freeze native full-prefill comparator/backend/dtype/masks/positions, controls, boundaries and input fixtures before core execution. Keep optional cached acceptance in a distinct protocol. Retain minimal sufficient lossless arrays: deduplicate identical immutable arrays by content identity with logical manifests and verified resolution; do not drop rows required for independent comparison or cast down precision. Reuse the existing guard first in a clean third-phase output directory and measure small-calibration guard overhead. Per-call whole-phase rescanning is a suspected, not isolated, cause of the prior 257-second/389-call runtime. Change guard accounting only if measured overhead justifies it; compare before/after, preserve enforcement, and validate accounting/reconciliation/interruption recovery independently. All new bulk and synthetic outputs use EXTERNAL_ARTIFACT_ROOT, never the constrained root filesystem.

Run producer, preserve receipt, then separate independent exact-candidate repeat. Scientific discovery/calibration fixtures and held-out fixtures remain separate. Freeze hypothesis and controls before held-out access. If eligibility fails, preserve failure; investigate using discovery data and a new preregistered held-out set rather than reusing exposed test cases. Produce actual trace and restoration evidence. Include a bounded candidate-versus-native intervention-equivalence check where the selected boundary contract permits it; baseline candidate parity alone does not prove intervention equivalence. If intervention is native-only, disclose that limitation explicitly and retain the core editable-candidate parity evidence. Independent verifier recomputes metrics from lossless raw data and actually repeats the required run.

## Phase 3 — integration and stopping

Bind atlas, map, source, runtime, protocols, trace, scientific evidence, tool reuse matrix and report into exact manifest; independently review. Stop at S0–S5 acceptance, reporting third-model scope and earlier Pythia strict-T2 failure. Further optional cache work cannot prolong an otherwise completed core task by default.

## Ownership and recovery

State/planning owner: third_scope_executor for initial packet only. Research workers own research subdirectory. Coordinator assigns acquisition, candidate, oracle/runtime, science and independent verifier with disjoint write boundaries. TASK_STATE.md is sole live status. Save every packet identity, reservation, terminal receipt and failure before next packet. A new worker reconciles ledger/process/artifact state before retries; no timer reset or erased charge. No runtime process is started by this planning packet.

## Methods research received; target selection still pending

Coordinator supplied evidence/transfer-003/research/methods-v1.md identity cfa4a4a57be975d95bb681554f34088f11e2114629c259ef5e9e220ea4e1aa8c (integration review pending). Preferred bounded hypothesis interface is head output before output projection, after validating native hook/layout for the chosen architecture. Preserve native packed-projection arithmetic. Apply identical edits to native and candidate and measure intervention parity. Freeze primary contrasts; secondary raw-logit gap, KL and paired bootstrap may be recomputed from retained logits without additional forwards. No positive-head result is guaranteed; a qualified controlled negative remains valid. TransformerLens, NNsight and pyvene are source-reviewed methods, not installed or validated integrations; no framework installation is required absent a demonstrated capability need.

## Initial concrete packet v2

See runtime-protocol.md, numerical-protocol.md and contracts/candidate.md. Research source identity: model-selection-v1.json SHA256 2cbb7adfb67abe0499f45d3b64b058a6f6fafd8234445493bb84b09c7223d65d. Remote safetensors size 548105171 bytes; SHA256 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707. Eight-file bundle 550967953 bytes; these are upstream expectations, not local acquisition proof. Source candidate and synthetic build may proceed; this scope executor performs no target load, download or installation. Separate acquisition admission is recorded below. Selection verifier PASS is reported; the separately assigned acquisition must still pass its local artifact audit before target loading.

Coordinator-reported independent research PASS receipt SHA256 172e95ce2573e150726a7116a4e61c4a43d4fcc9e7c75a9f0b0c2ca2ad7cce40. Acquisition is assigned separately at EXTERNAL_ARTIFACT_ROOT/workspaces/model-transfer-003-gpt2/artifacts/original; no local acquisition completion is asserted here.
