# Independent selection verification

PASS_ACQUISITION_ONLY — 2026-11-18T19:04:06.246505+00:00

13 checks passed: proposal_hash, revision, public_ungated, config_exact, card_apache, implementation_apache, no_auto_map, allowlist_sizes, allowlist_sum, weight_lfs, parameters, installed_sources, installed_version.

Candidate: EleutherAI/pythia-14m @ cf967c0a9a04383db6f7b1108d86b2962634b4ac. Seven allowlisted files total 30,274,606 bytes; checkpoint 28,143,920 bytes.

- Packed head-local QKV must not be split as three global matrices.
- Partial RoPE covers 8 of 32 head dimensions; preserve unrotated dimensions.
- Parallel residual and LayerNorm/GELU differ meaningfully from MiniMind.
- 50304 output rows need not equal tokenizer vocabulary; independently audit vocabulary gap.
- CPU feasibility is size-based inference only; float32 weights approximately 56.3 MB, runtime overhead unmeasured.
- License attribution is upstream declaration; card discourages human-facing deployment.
- No checkpoint downloaded, loaded, executed, or hashed locally. No runtime admission.

Sources:
- https://huggingface.co/api/models/EleutherAI/pythia-14m/revision/cf967c0a9a04383db6f7b1108d86b2962634b4ac?blobs=true
- https://huggingface.co/EleutherAI/pythia-14m/resolve/cf967c0a9a04383db6f7b1108d86b2962634b4ac/README.md
- https://huggingface.co/EleutherAI/pythia-14m/resolve/cf967c0a9a04383db6f7b1108d86b2962634b4ac/config.json
- https://huggingface.co/EleutherAI/pythia-14m/resolve/cf967c0a9a04383db6f7b1108d86b2962634b4ac/tokenizer_config.json
- https://raw.githubusercontent.com/huggingface/transformers/v5.1.0/LICENSE
- PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox
