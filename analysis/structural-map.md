# Tensor-to-computation hypothesis v1

91 stored tensors mapped; graph hypotheses are prior-informed, not blind discoveries. Constants are evaluator-supplied.

- input IDs -> embedding lookup; status: prior-informed. Alternatives: Embedding matrix orientation inferred from dimensions.
- 8 sequential pre-normalized residual blocks; status: prior-informed. Alternatives: Order and residual connections not uniquely determined by weights.
- RMSNorm -> Q/K/V linears -> per-head Q/K RMSNorm -> split-half RoPE; status: prior-informed. Alternatives: Normalization placement uncertain until oracle check.
- 8 query heads, 4 KV heads, head dimension 96; repeated grouped keys/values; status: prior-informed. Alternatives: Other head permutations fit stored shapes.
- causal scaled dot attention -> output projection -> residual add; status: prior-informed. Alternatives: Masking conventions supplied.
- RMSNorm -> SiLU(gate) times up -> down -> residual add; status: prior-informed. Alternatives: Activation supplied; ordering prior-informed.
- final RMSNorm -> vocabulary logits; status: prior-informed. Alternatives: Output projection numerical tie not assumed.

All tensors individually mapped in structural-map.json. Unique graph recovery, training history and semantic circuit claims remain unsupported.
