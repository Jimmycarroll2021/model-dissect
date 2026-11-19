# Applying the model-dissection workflow across three models

**Three applications have their required scientific evidence: MiniMind, the explicitly scoped Pythia full-prefill slice, and GPT-2. GPT-2 final delivery audit/controller closure is separate and tracked in task state.**

## Established applications and scope

| Application | Established status | Evidence boundary |
|---|---|---|
| MiniMind | Original application completed | Checkpoint/backend-specific source-assisted result; see [workbench](reverse-engineering-workbench.md) and [cached-softmax matrix report](cache-softmax-matrix-result.md). No universal numerical-equality claim. |
| Pythia-14M | Full-prefill scientific/process-transfer slice independently completed | Strict cached/full T2 remains FAILED. [Final review](../evidence/transfer-002/delivery/final-review-v1.json), SHA256c4c667fe542b5102f2e817e7aed694caa157a264b62bace977dca3b74632321c, accepts the scoped delivery, not the original strict whole objective. See [result](pythia-transfer-result.md). |
| GPT-2 | Full numerical, atlas, trace and scientific independent repeats PASS; final delivery audit separate |16fixtures/15boundaries/1,680comparisons per role, zero error and1,440byte-identical arrays in [full review](../evidence/transfer-003/review/full-review-v2.json). Atlas actual repeat also PASS; qualified REFUTED science independently repeated; final delivery audit separate. |

## Workflow a user can inspect and repeat

Start with the output you need to inspect: every stored tensor and its role, an executable candidate, a concrete input trace, and a falsifiable intervention. Pin checkpoint/config/tokenizer/source identities and choose a declared mode before testing. Identify buffers and ties separately from learned weights. Preserve exact originals and prior failed results.

Build the editable computation separately from the known native oracle. Label source exposure honestly. Check a small real run first, including lossless serialization/reload, restoration, resource use and all numerical boundaries; then admit the larger frozen matrix. Require an independent executor to repeat the exact candidate and recompute raw evidence. Save controls and failed attempts alongside positive results.

Choose internal sites using development data only. Freeze held-out contrasts, eligibility, controls and thresholds before evaluation. Apply the same actual edits to native and candidate; baseline parity alone cannot establish intervention parity. Report qualified negative findings without replacing the hypothesis after seeing held-out outcomes. Insufficient qualified data is unfinished evidence, not a negative scientific result.

Finish with exact manifest review and a report stating what transferred unchanged, what required an adapter, and which claims remain unsupported. Stop after accepted outcomes; do not turn optional cache/backend exploration into an unbounded completion gate.

## Failure -> consequence -> replacement

| Earlier observation | Consequence | Applied replacement |
|---|---|---|
| Pythia cross-actor equality coexisted with failed cached/full parity | A global cache gate blocked useful scoped investigation | Preregister GPT-2 native full-prefill core and optional cached acceptance separately; retain old failure unchanged. |
| Primitive repairs did not preserve every native anchor | A matching local operation was insufficient evidence | Test singleton/long inputs, downstream boundaries and native anchors before promotion. |
| Large raw output and repeated guard scans can be expensive | Storage and verification cost can dominate small experiments | Measure real serialization and guard overhead in a clean phase; retain minimal sufficient lossless arrays. Guard overhead remains a hypothesis without isolated timing comparison. |
| Source methods and architecture seams differ | A valid NeoX hook name would be wrong for GPT-2 | Validate GPT-2 c_proj preprojection layout against observed attention-weighted-V and preserve packed arithmetic. |
| Tiny-model qualification can fail | All-ineligible data cannot test the hypothesis | Freeze deterministic baseline eligibility and require adequate held-out cases; preserve incomplete outcomes. |
| Synthetic build checks can be mistaken for target proof | Readiness claims outrun observed behavior | Keep artifact audit, synthetic build, actual pilot, full repeat and scientific acceptance as distinct evidence stages. |

## Provisional reuse matrix

| Component | Reuse decision | Current limit |
|---|---|---|
| Spec Kit state, identities and independent-review process | Reuse workflow | Markdown does not enforce runtime isolation or automatic recovery. |
| Bounded phase guard, packet accounting and lossless evidence | Reuse with new phase identity and reviewed caps | Small pilot measured; no generic performance improvement claim. |
| Tensor inventory and evidence-labelled graph | Reuse schema; GPT-2 adapter distinguishes parameters/buffers/ties | Actual atlas independently repeated; population std derived from rounded saved moments, not centered weight reduction. |
| Editable numerical computation | New GPT-2 adapter: learned positions, Conv1D and tied readout | Frozen16-fixture full numerical matrix independently repeated; arbitrary inputs/cached/left-padding unverified. |
| Head interventions/orthogonal controls/secondary metrics | Reuse method with validated GPT-2 seam and actual candidate edits | Synthetic science build reviewed; tokenization admitted; qualified REFUTED independently repeated with actual candidate/native edits, controls and trace audit. |
| TransformerLens/NNsight/pyvene ideas | Source-reviewed methodological references | No installed/validated integration claimed. |

Primary-source context and exact upstream references are collected in [methods research](../evidence/transfer-003/research/methods-v1.md) and [model selection](../evidence/transfer-003/research/model-selection-v1.md). The [GPT-2 provisional report](gpt2-transfer-result.md) separates current measurements from remaining work. Final completion requires exact delivery-manifest review; scientific acceptance receipts do not substitute for that final gate.

Latest atlas evidence: [actual repeat review](../evidence/transfer-003/review/atlas-actual-v2.json) and [derived population-std supplement](../evidence/transfer-003/static-supplement/producer-population-std-v1.json). Exact repeated matrix output plus reviewed orientation does not claim a separately implemented weight-level arithmetic oracle. Tokenization was independently decoded from saved IDs, not independently re-encoded. Independent scientific repetition and acceptance are now established by science-review-v2.json; final delivery acceptance remains a distinct gate.

## Repeated scientific finding and improved evidence boundary

GPT-2's development-selected layer8/head8 transferred a small positive contrast: independently repeated median normalized recovery0.08404521678273584 and orthogonal-control advantage0.09451938285766666, with16/16positive eligible held-out pairs. The frozen medians required0.20and0.10, so the independently verified verdict is qualified REFUTED. This is not a statement that the head is useless; the result is bounded to this site, corpus, contrast and thresholds. [Result and chart](gpt2-transfer-result.md) retain the exact criteria and independent-review status.

A material workflow improvement is testing identical actual head edits in both native and editable candidate, with full retained readout logits and complement/restoration evidence. Baseline candidate parity alone cannot establish intervention parity. Independent actual repeat and standalone raw/trace/control/metric checks now pass. A diagram and metric-only PNG/SVG chart make the outcome inspectable without adding model calls. No measured delivery-speed or universal-automation improvement is claimed.

## Model comparison with distinct scientific claims

| Model and established size | Completed evidence scope | Scientific result | Material limit |
|---|---|---|---|
| MiniMind,68,827,392stored values across91FP16tensors | Source-assisted map, editable numerical mode, actual trace and independent experiment/reconstruction repeat | Layer6/head2 echoed-digit transfer SUPPORTED_BOUNDED_HEAD_MEDIATION;16/16eligible;median minimum-direction recovery0.5534745383939315 | Stored values do not establish independent learned-parameter count; development-selected weak same-layer control; no complete copying circuit/necessity claim. |
| Pythia-14M,14,067,712stored parameters | Accepted full-prefill numerical/static/scientific slice and independent repeat | Block4 SUPPORTED_BOUNDED_BLOCK_EFFECT;16/16eligible;median minimum-direction recovery0.7104862948449367 | Strict cached/full T2 remains FAILED; whole-block effect and narrow template holdout, not head/circuit recovery. |
| GPT-2,124,439,808learned parameter elements plus12serialized causal buffers | Full-prefill matrix, atlas, actual attention trace and actual native/candidate head edits independently repeated | Layer8/head8 qualified REFUTED;16/16eligible;median mean-direction recovery0.08404521678273584 below0.20 | Frozen prompts/mode only; median advantage0.09451938285766666 below0.10 despite16/16positive pairs; not head uselessness. |

These recovery metrics are not directly comparable effect-size rankings: MiniMind/Pythia use minimum bidirectional recovery, GPT-2 uses mean bidirectional recovery; contrasts, interventions, controls and prompt distributions differ. MiniMind finding: [independent receipt](../evidence/echo-mechanism/review/recovery-independent-final-v1.json) and [report](echo-mechanism-result.md). Pythia finding: [independent receipt](../evidence/transfer-002/review/science-full-independent-v2-results.json). GPT-2 finding: [independent receipt](../evidence/transfer-003/review/science-review-v2.json). Historical failures and their evidence remain unchanged.
