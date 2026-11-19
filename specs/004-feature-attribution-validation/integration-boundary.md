# Existing harness reuse and missing feature seam

This is a read-only design assessment, not an implemented integration or new runtime validation.

Reviewed local sources: `docs/gpt2-transfer-result.md`, `specs/003-model-dissection/plan.md`, `specs/003-model-dissection/internal-protocol-v1.md`, `evidence/transfer-003/science/protocol-v1.json`, and the interface/edit portions of `src/minimind_re/transfer3/science.py`.

| Existing evidence/practice | Reuse in this pilot | Boundary |
|---|---|---|
| Exact model/tokenizer/source identities and lossless receipts | Freeze graph, feature release, original model and all intervention inputs | GPT-2 identity does not identify another hosted model or dictionary |
| Native/candidate parity and complete retained logits | Require faithful original execution and output evidence | A replacement model is an approximation; its reconstruction error needs a distinct gate |
| Development/held-out split and immutable thresholds | Separate graph discovery from confirmation and lock before effect probes | The prior baseline-derived output contrast is not automatically the new semantic target |
| No-op, restoration, complement invariance and control matching | Adapt checks to feature coefficient/decoder contribution at exact positions | Existing head-vector check does not validate feature interception |
| Actual independent repeat and standalone metrics | Repeat exact feature candidate and recompute all checks | Prior acceptance is not feature-level acceptance |
| Guard/receipt/budget patterns | Reuse the accounting design after compatibility review | Prior runtime admission/caps are not new execution authority |

The existing `transfer3/science.py` interface uses `head_transform`; the edit path checks `[1,S,12,64]` FP32 head tensors and writes `heads[0,-1,h]`. That is a concrete GPT-2 head seam. This inspection does not establish an SAE/transcoder feature encoder, decoder, replacement-model adapter or hosted intervention/export implementation. None was added here.

A feature implementation would still need verified dictionary loading/identity, encoder/decoder conventions, feature-token indexing, reconstruction/error-term policy, original/replacement comparison, feature coefficient capture/edit, matched controls and hosted or local raw evidence export. For a different model the current hard-coded head shape and numerical contracts must not be silently reused. The graph discovery interface and causal intervention engine are separate capabilities.

The prior report records a qualified REFUTED head-transfer result. Preserve that result; it neither refutes this new feature claim nor licenses threshold tuning. Its stored scientific thresholds and CPU resource usage belong to that exact experiment.

Spec Kit skill adaptation: the parent's explicit ownership limits exclude `.specify/feature.json`, root pointers, accepted 003 documents and branch changes. This leaf therefore uses the specification template but leaves root workflow selection to its owner. No extension configuration was found at `.specify/extensions.yml`. The new feature is a draft, not ready for causal execution or final plan acceptance.
