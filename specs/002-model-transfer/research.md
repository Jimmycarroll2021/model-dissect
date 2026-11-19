# Design decisions and unresolved model binding

Decision: transfer the evidence chain, not the MiniMind architecture. Rationale: accepted whole-objective v2 requires both independent computation and controlled observations. Rejected alternative: an inventory plus native-forward wrapper cannot meet T2/T4.

Decision: one thin architecture module with separate native reference, retaining legacy code. Rationale: inventory hashing/statistics are portable, while RMS/RoPE/GQA/gated MLP, hook paths and hardcoded admissions are not. Rejected alternative: broad plugin framework before a second demonstrated adapter.

Decision: one aggregate guarded phase. Rationale: avoids repeated guard clones and independently overspendable worker budgets; keeps failed reservations durable.

Pending selection worker: exact model/revision/license, artifact sizes/format, config/tokenizer compatibility, native source revision, environment dependency feasibility, context/local-attention constraints. This is a design draft, not a research-complete runtime packet. Bind those facts before implementation-specific protocol review.

Evidence consulted: current spec001 plan/quickstart, constitution, workbench, whole-objective-review-v2, inventory.py, atlas.py, probe.py, mechanism/matrix signatures. Earlier memory was routing context only; current acceptance read directly.

## Proposed model binding (not independently verified here)

Selection source: ../../evidence/transfer-002/selection-proposal.json. EleutherAI/pythia-14m, revision cf967c0a9a04383db6f7b1108d86b2962634b4ac; Apache-2.0; model.safetensors 28,143,920 bytes, expected SHA256 116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1; seven-file allowlist 30,274,606 bytes. Metadata reports 14,067,712 parameters. Installed proposal environment: torch 2.11.0+cu128, transformers 5.1.0, tokenizers 0.22.2, safetensors 0.8.0rc0. Loading remains untested.

GPT-NeoX config: six blocks, hidden128, four heads, intermediate512, parallel residual, LayerNorm epsilon1e-5, GELU, untied embeddings, biased attention, rotary fraction0.25, max positions2048, vocab50304. Coordinator reports tokenizer50277/no pad; verify tokenizer bytes at acquisition. Preserve/account all50304 input/output rows, including unmapped tokenizer IDs; never crop logits to decoder vocabulary. Explicitly freeze padding fill IDs, logical positions and mask semantics; no invented pad-token mutation. Model format stores FP16; declared reconstruction/reference execution casts consistently to FP32. Native eager reference and independent primitive candidate remain separate.

No useful generation competence is assumed. A distributional intervention hypothesis can be valid without successful answers, provided baseline contrast eligibility/denominators and damage/control criteria are preregistered. Unsupported/refuted/inconclusive completed controlled experiments may satisfy T4; zero eligible observations or unexecuted arms cannot.

## Selection verification update
Read independent receipt ../../evidence/transfer-002/review/selection-v1.json, SHA256 ee6e2b2580a675aea3e5bf75a430800beff5f1bb968e958b79b9c7d95527821b: PASS_ACQUISITION_ONLY,13checks. Thus model selection is confirmed for acquisition, not model loading or runtime. Coordinator reports download in progress at EXTERNAL_ARTIFACT_ROOT/workspaces/model-transfer-002-pythia-14m/artifacts/original; this planning worker has not verified acquisition completion. Earlier pending paragraphs above are drafting history superseded by this update.
