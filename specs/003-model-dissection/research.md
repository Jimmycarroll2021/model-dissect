# GPT-2 selection integration

Decision: provisionally select openai-community/gpt2 revision 607a30d783dfa663caf39e06633721c8d4cfcd7e; independent selection PASS; acquisition audit pending.
Rationale: 124M advertised scale is somewhat larger than original MiniMind and substantially larger than Pythia-14M; learned absolute positions, Conv1D projections and tied output exercise different architecture adapters.
Alternatives: Pythia-70M reuses the previous architecture; SmolLM2-135M has 30 layers and more hook surface. Exact comparison and primary references: evidence/transfer-003/research/model-selection-v1.json and model-selection-v1.md. Methods: evidence/transfer-003/research/methods-v1.md.
Upstream bundle: 550967953 bytes, of which safetensors 548105171 bytes, expected SHA256 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707. Inventory learned parameters, buffers and tied aliases independently; stored bytes are not parameter-count proof.
No model acquisition/runtime or optional framework installation follows from research alone. Native source assistance must be disclosed. Historical cached/full failures remain unchanged.

Coordinator-reported independent research PASS receipt SHA256 172e95ce2573e150726a7116a4e61c4a43d4fcc9e7c75a9f0b0c2ca2ad7cce40. Acquisition is assigned separately at EXTERNAL_ARTIFACT_ROOT/workspaces/model-transfer-003-gpt2/artifacts/original; no local acquisition completion is asserted here.
