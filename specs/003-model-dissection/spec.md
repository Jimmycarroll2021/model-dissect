# Feature specification: third model dissection

Status: PRELIMINARY — GPT-2 selection independently reviewed; initial protocol/build review pending.
Date: 2026-11-19. Feature: 003-model-dissection.

## User scenarios and testing

### US1 — inspect a third model (P1)
Jimmy can inspect every stored weight and an evidence-labelled computation map for a somewhat larger open-source model. Acceptance: exact immutable source, checkpoint, configuration and tokenizer identity; exhaustive tensor accounting including aliases/ties; each graph assertion labelled observed, source-assisted, inferred or unverified. Unmapped tensors prevent acceptance.

### US2 — execute and inspect faithful computation (P1)
Jimmy can inspect an editable independently composed candidate and follow an actual input through it. Acceptance: candidate versus separately operated native reference agrees under a frozen full-prefill protocol; trace includes named boundaries, shapes and numerical samples bound to retained arrays. A separate executor repeats the exact candidate independently. Wrapping the native forward is not reconstruction.

### US3 — test a bounded internal hypothesis (P1)
Jimmy can inspect one qualified controlled intervention and its independent repeat. Acceptance: freeze hypothesis, clean/corrupt contrasts, no-op, matched internal control, restoration, metrics, eligibility and held-out split before held-out execution. Use 16 qualified held-out cases by default, or justify and preregister another count before execution. Adequately tested negative results are valid; all-ineligible or underpowered data do not establish completion.

### US4 — understand reuse and stop after three applications (P2)
Jimmy receives a reproducible report and tool reuse matrix naming unchanged components, adapters, limitations and avoided failures. The third application closes only after all required third-model gates pass. Stop after three successful scoped applications: original MiniMind complete; Pythia-14M full-prefill slice complete while strict cached T2 FAILED; third model pending. Do not relabel Pythia strict failure or claim universal model recovery.

## Functional requirements

- FR01: Preserve prior records and original model artifacts with exact identities.
- FR02: Select an open-source somewhat larger model using primary repository/paper evidence, licence, native oracle availability and measured feasible resources.
- FR03: Produce exhaustive weight atlas and graph map with source exposure disclosed.
- FR04: Deliver editable source-assisted reconstruction with mandatory native-faithful full-prefill parity preregistered from the outset.
- FR05: Treat cached decoding/backend portability as a separate optional acceptance track; failure cannot silently change core thresholds or be described as passing.
- FR06: Save lossless minimal sufficient raw evidence for independent recomputation, provenance, no-op/restoration and actual trace; test real serialization and reload.
- FR07: Freeze scientific criteria before held-out evaluation; preserve failed, negative and ineligible results.
- FR08: Independently repeat numerical and scientific requirements against the exact candidate and bind final deliverables to an immutable manifest.
- FR09: Bound acquisition, positions, elapsed/process time, memory, disk and concurrency; preserve unrelated workloads and fail closed when capacity is insufficient.
- FR10: Retain failures and lessons; stop optional expansion once required acceptance passes.

## Success criteria

S0: Identity and preservation independently verified; runtime admitted only after fresh resource checks.
S1: 100% stored tensors accounted for, with all computation claims evidence-labelled.
S2: Every frozen core full-prefill parity case passes native fidelity, no-op and restoration; independent actual repeat passes.
S3: At least one actual trace reproducibly links input through internal operations to output.
S4: One preregistered intervention has adequate qualified held-out evidence and independent repeat, with bounded positive, negative or inconclusive scientific interpretation. An inconclusive result caused by inadequate qualified data does not meet S4.
S5: Final exact-manifest audit passes and reuse matrix/report preserves scope limitations.

## Edge cases and assumptions

Unknown/tied tensors, missing source, tokenizer mismatches, nonfinite rows, failed serialization, shape-dependent rounding, insufficient eligible cases and resource exhaustion produce saved explicit failures. No training, model improvement, broad quantization, service changes, publication or cleanup is in this task. Pinned GPT-2 identity and initial numerical/resource protocols are recorded in plan.md and runtime-protocol.md; fresh fixture and scientific protocol freeze remain prerequisites to dependent execution.
