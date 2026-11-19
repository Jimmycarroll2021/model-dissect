# Initial candidate/reference contract

Build contract pending integration review; source implementation may refine naming before fixture freeze.

Candidate.from_directory(directory, expected_config_sha256, expected_weights_sha256) binds original identities and rejects mismatches before loading. forward(input_ids, attention_mask, position_ids, use_cache=False, boundary_hook=None, head_hook=None) returns complete logits plus named captured boundaries; cache/past requests rejected in core mode. Callers supply explicit mask/positions. Hooks must not mutate original weights or saved input arrays. Head hook receives layer index and [B,S,12,64] logical head values before output projection and returns same shape/dtype; implement validated view conversion without changing projection arithmetic. Boundary hook names embedding, block0 through block11, final_norm, logits. Record token/mask/position arrays and all hook configuration identities.

Native oracle implements same external evidence contract independently using installed GPT2LMHeadModel eager execution. Candidate mathematical helpers cannot be shared with oracle. Model-free synthetic tests cover Conv1D orientation, causal/additive masks, position embedding, tied logits, packed projection head layout, hook no-op/restoration, unsupported-cache rejection and lossless real serialization.

Source/build owner owns only new third-model candidate files; oracle/runtime owner and verifier are separate. Exact source identity and any refined API enter integrated manifest before runtime review.

GPT-2 native seam is transformer.h[L].attn.c_proj input, not the NeoX seam discussed conditionally in methods research. Native input [B,S,768] must be validated then reshaped [B,S,12,64]; coordinator-supplied installed source identity c05a34ca83eecf3c9c84627e4461b544b4628cf27c764509578710d8dd5f03ca requires build owner verification.

Capability contract: support unpadded and right-padded full-prefill with first key visible; reject left padding, all-masked or first-key-hidden inputs explicitly. Expose capabilities in manifest; native reference uses exactly the same supported masks/positions. Intervention and readout index last real token, with clean/corrupt alignment checked explicitly.
