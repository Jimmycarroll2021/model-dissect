# GPT-NeoX transfer computation map

This is a **source-assisted static map**, derived from the verified Pythia-14m configuration, admitted safetensors header and inspected installed Transformers 5.1.0 implementation. It does not demonstrate numerical parity, actual token behavior or causal mechanism. Candidate and independent reference runtime results remain separate gates.

The selected checkpoint is EleutherAI/pythia-14m at cf967c0a9a04383db6f7b1108d86b2962634b4ac. Its configuration gives six blocks, hidden width 128, four heads of width 32, MLP width 512, vocabulary width 50304, and 8 rotary dimensions per head. Operator code derives these sizes from configuration rather than fixing checkpoint-specific dimensions.

For each block, input LayerNorm feeds a single head-packed QKV affine projection. The 384 projected coordinates reshape as [4 heads, 3 Q/K/V groups, 32 dimensions]. Thus each head occupies 96 consecutive output coordinates, with its Q, K, V occupying consecutive 32-coordinate slices. Only the first eight dimensions of Q and K receive default split-half RoPE. K/V cache stores rotated keys and unrotated values with physical padding slots retained. Logical positions and full historical key masks are explicit for padded calls.

Eager attention computes QK^T / sqrt(32), adds causal and key-padding masks, uses native FP32 softmax, multiplies V, concatenates heads and applies the attention output affine. A separate LayerNorm of the same incoming residual feeds MLP expansion, exact GELU and contraction. The branches combine in native floating-point order: **MLP output + attention output + incoming residual**. Final LayerNorm precedes the untied, bias-free full-vocabulary output projection. Both LayerNorms within each block and the final LayerNorm have learned scales and offsets. All attention/MLP affine biases appear explicitly below.

Input embedding and output projection are distinct tensors. All 50304 rows contribute to full-tensor statistics. The verified tokenizer vocabulary has 50277 assigned entries; the remaining 27 model rows remain present, including all output logits. Static vocabulary views use fixed numeric IDs 0 through 127 and do not claim semantic coverage.

## Source-cited model equations

Use row vectors and stored PyTorch weights in `[output, input]` order. Suppress the batch axis; `T` denotes physical sequence length, `H=128`, `A=4` heads, `D=H/A=32`, `I=512`, and `V=50304`. All broadcasts below are over token rows. The equations define mathematical roles; the candidate uses native primitives and the operation order described above rather than treating algebraic reassociation as numerically interchangeable.

**Embedding.** Let `E = gpt_neox.embed_in.weight ∈ R^(V×H)`. For token ID `t_s`, the initial residual row is `X⁰_s = E[t_s,:]`, so `X⁰ ∈ R^(T×H)`. There is no learned absolute-position embedding tensor. Installed embedding and model construction source (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:404`; external or omitted historical artifact).

**Every LayerNorm scale and bias.** For a hidden row `x ∈ R^H`, define

```text
μ(x) = (1/H) Σ[j=0..H−1] x_j
σ²(x) = (1/H) Σ[j=0..H−1] (x_j − μ(x))²
LN(x; γ, β)_j = ((x_j − μ(x)) / sqrt(σ²(x) + ε)) γ_j + β_j
ε = config.layer_norm_eps = 1e−5
```

The variance uses denominator `H` (correction zero). For each block prefix `p = gpt_neox.layers.ℓ.`, `γ₁,β₁` are `p+input_layernorm.weight/bias`, and `γ₂,β₂` are `p+post_attention_layernorm.weight/bias`, each shape `[H]`. Final `γ_f,β_f` are `gpt_neox.final_layer_norm.weight/bias`, also `[H]`. Native LayerNorm definition (`PRIVATE_HOME/.local/lib/python3.12/site-packages/torch/nn/modules/normalization.py:106`; external or omitted historical artifact); GPT-NeoX layer construction (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:255`; external or omitted historical artifact).

**Packed QKV affine and head slices.** For block input `X ∈ R^(T×H)`:

```text
N₁ = LN(X; γ₁, β₁)
Z = N₁ W_qkvᵀ + b_qkv                 [T, 3H]
W_qkv = p+attention.query_key_value.weight   [3H, H] = [384,128]
b_qkv = p+attention.query_key_value.bias     [3H]    = [384]
Q_a = Z[:, 3aD       : (3a+1)D]        [T,D]
K_a = Z[:, (3a+1)D   : (3a+2)D]        [T,D]
V_a = Z[:, (3a+2)D   : (3a+3)D]        [T,D],  a ∈ {0,1,2,3}
```

These slices are per-head packed, not globally contiguous Q/K/V thirds. GPT-NeoX attention projection and split (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:189`; external or omitted historical artifact).

**Partial split-half RoPE.** Set `R = int(D×rotary_pct) = 8`, `base=10000`, and `ω_j=base^(−2j/R)` for `j=0..3`. At explicit logical position `u`, define `θ=(uω₀,uω₁,uω₂,uω₃,uω₀,uω₁,uω₂,uω₃)`. For either Q or K row `q`, write `r=q[:8]=(r₀,…,r₇)` and `rotate_half(r)=(-r₄,-r₅,-r₆,-r₇,r₀,r₁,r₂,r₃)`:

```text
rotated(q,u) = concat(r ⊙ cos(θ) + rotate_half(r) ⊙ sin(θ), q[8:32])
Q̃_a[s,:] = rotated(Q_a[s,:], position_ids[s])
K̃_a[s,:] = rotated(K_a[s,:], position_ids[s])
```

Only Q/K coordinates 0..7 rotate; coordinates 8..31 and all V coordinates pass through unchanged. Frequencies are derived, not checkpoint parameters. Cached keys are already rotated. Frequency construction (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:52`; external or omitted historical artifact); split-half rotation and partial concatenation (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:119`; external or omitted historical artifact).

**Attention and output affine.** With current query length `T_q` and cached-plus-current key length `T_k`:

```text
S_a = (Q̃_a K̃_aᵀ) × D^(−1/2) + M      [T_q,T_k]
P_a = softmax_FP32(S_a, key_axis)       [T_q,T_k]
C_a = P_a V_a                          [T_q,D]
C = concat_heads(C₀,C₁,C₂,C₃)          [T_q,H]
Attention = C W_oᵀ + b_o              [T_q,H]
W_o = p+attention.dense.weight         [H,H]
b_o = p+attention.dense.bias           [H]
```

`M` is zero for causal, unmasked physical keys and FP32 finite minimum for excluded keys. Historical masks cover every cached physical slot. Eager evaluation dropout is inactive. Eager attention ordering (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:162`; external or omitted historical artifact); attention output projection (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:189`; external or omitted historical artifact).

**Parallel MLP and residual.** Both normalization branches consume the same incoming `X`:

```text
N₂ = LN(X; γ₂, β₂)                    [T_q,H]
B = N₂ W_upᵀ + b_up                  [T_q,I]
GELU(z) = 0.5 z (1 + erf(z / sqrt(2)))
MLP = GELU(B) W_downᵀ + b_down        [T_q,H]
X_next = (MLP + Attention) + X        [T_q,H]
W_up   = p+mlp.dense_h_to_4h.weight    [I,H] = [512,128]
b_up   = p+mlp.dense_h_to_4h.bias      [I]   = [512]
W_down = p+mlp.dense_4h_to_h.weight    [H,I] = [128,512]
b_down = p+mlp.dense_4h_to_h.bias      [H]   = [128]
```

Exact native GELU is used, not the tanh approximation. The residual parentheses record the native addition order. Repeat for blocks `ℓ=0..5`. MLP affine sequence (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:38`; external or omitted historical artifact); GELU dispatch (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/activations.py:70`; external or omitted historical artifact); parallel residual (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:255`; external or omitted historical artifact).

**Final normalization and unembedding.** After six blocks, `F=LN(X⁶;γ_f,β_f) ∈ R^(T_q×H)`. Let `U=embed_out.weight ∈ R^(V×H)`, stored separately from `E`. The complete output is `logits=F Uᵀ ∈ R^(T_q×V)`, with no output bias and no discarded vocabulary rows. Final LayerNorm (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:494`; external or omitted historical artifact); untied bias-free output construction (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:527`; external or omitted historical artifact); logit projection (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:594`; external or omitted historical artifact).

The tensor-name suffixes above bind every row of the exhaustive table below to an equation: 12 tensors per block × 6 blocks, plus input embedding, output projection and final LayerNorm scale/bias = 76. These model equations do not remove the omissions in the static QK/OV surrogates documented later.

## Exhaustive tensor-to-operation map

The following 76 entries cover every header key exactly once; shapes are the observed configuration-derived dimensions.

| Tensor key | Shape | Operation role |
|---|---|---|
| `embed_out.weight` | 50304 × 128 | untied vocabulary output projection rows |
| `gpt_neox.embed_in.weight` | 50304 × 128 | input token embedding rows |
| `gpt_neox.final_layer_norm.bias` | 128 | LayerNorm offset |
| `gpt_neox.final_layer_norm.weight` | 128 | LayerNorm scale |
| `gpt_neox.layers.0.attention.dense.bias` | 128 | attention output bias |
| `gpt_neox.layers.0.attention.dense.weight` | 128 × 128 | attention output projection |
| `gpt_neox.layers.0.attention.query_key_value.bias` | 384 | head-packed [head,Q/K/V,dimension] bias |
| `gpt_neox.layers.0.attention.query_key_value.weight` | 384 × 128 | head-packed [head,Q/K/V,dimension] projection |
| `gpt_neox.layers.0.input_layernorm.bias` | 128 | LayerNorm offset |
| `gpt_neox.layers.0.input_layernorm.weight` | 128 | LayerNorm scale |
| `gpt_neox.layers.0.mlp.dense_4h_to_h.bias` | 128 | post-GELU contraction bias |
| `gpt_neox.layers.0.mlp.dense_4h_to_h.weight` | 128 × 512 | post-GELU contraction projection |
| `gpt_neox.layers.0.mlp.dense_h_to_4h.bias` | 512 | pre-GELU expansion bias |
| `gpt_neox.layers.0.mlp.dense_h_to_4h.weight` | 512 × 128 | pre-GELU expansion projection |
| `gpt_neox.layers.0.post_attention_layernorm.bias` | 128 | LayerNorm offset |
| `gpt_neox.layers.0.post_attention_layernorm.weight` | 128 | LayerNorm scale |
| `gpt_neox.layers.1.attention.dense.bias` | 128 | attention output bias |
| `gpt_neox.layers.1.attention.dense.weight` | 128 × 128 | attention output projection |
| `gpt_neox.layers.1.attention.query_key_value.bias` | 384 | head-packed [head,Q/K/V,dimension] bias |
| `gpt_neox.layers.1.attention.query_key_value.weight` | 384 × 128 | head-packed [head,Q/K/V,dimension] projection |
| `gpt_neox.layers.1.input_layernorm.bias` | 128 | LayerNorm offset |
| `gpt_neox.layers.1.input_layernorm.weight` | 128 | LayerNorm scale |
| `gpt_neox.layers.1.mlp.dense_4h_to_h.bias` | 128 | post-GELU contraction bias |
| `gpt_neox.layers.1.mlp.dense_4h_to_h.weight` | 128 × 512 | post-GELU contraction projection |
| `gpt_neox.layers.1.mlp.dense_h_to_4h.bias` | 512 | pre-GELU expansion bias |
| `gpt_neox.layers.1.mlp.dense_h_to_4h.weight` | 512 × 128 | pre-GELU expansion projection |
| `gpt_neox.layers.1.post_attention_layernorm.bias` | 128 | LayerNorm offset |
| `gpt_neox.layers.1.post_attention_layernorm.weight` | 128 | LayerNorm scale |
| `gpt_neox.layers.2.attention.dense.bias` | 128 | attention output bias |
| `gpt_neox.layers.2.attention.dense.weight` | 128 × 128 | attention output projection |
| `gpt_neox.layers.2.attention.query_key_value.bias` | 384 | head-packed [head,Q/K/V,dimension] bias |
| `gpt_neox.layers.2.attention.query_key_value.weight` | 384 × 128 | head-packed [head,Q/K/V,dimension] projection |
| `gpt_neox.layers.2.input_layernorm.bias` | 128 | LayerNorm offset |
| `gpt_neox.layers.2.input_layernorm.weight` | 128 | LayerNorm scale |
| `gpt_neox.layers.2.mlp.dense_4h_to_h.bias` | 128 | post-GELU contraction bias |
| `gpt_neox.layers.2.mlp.dense_4h_to_h.weight` | 128 × 512 | post-GELU contraction projection |
| `gpt_neox.layers.2.mlp.dense_h_to_4h.bias` | 512 | pre-GELU expansion bias |
| `gpt_neox.layers.2.mlp.dense_h_to_4h.weight` | 512 × 128 | pre-GELU expansion projection |
| `gpt_neox.layers.2.post_attention_layernorm.bias` | 128 | LayerNorm offset |
| `gpt_neox.layers.2.post_attention_layernorm.weight` | 128 | LayerNorm scale |
| `gpt_neox.layers.3.attention.dense.bias` | 128 | attention output bias |
| `gpt_neox.layers.3.attention.dense.weight` | 128 × 128 | attention output projection |
| `gpt_neox.layers.3.attention.query_key_value.bias` | 384 | head-packed [head,Q/K/V,dimension] bias |
| `gpt_neox.layers.3.attention.query_key_value.weight` | 384 × 128 | head-packed [head,Q/K/V,dimension] projection |
| `gpt_neox.layers.3.input_layernorm.bias` | 128 | LayerNorm offset |
| `gpt_neox.layers.3.input_layernorm.weight` | 128 | LayerNorm scale |
| `gpt_neox.layers.3.mlp.dense_4h_to_h.bias` | 128 | post-GELU contraction bias |
| `gpt_neox.layers.3.mlp.dense_4h_to_h.weight` | 128 × 512 | post-GELU contraction projection |
| `gpt_neox.layers.3.mlp.dense_h_to_4h.bias` | 512 | pre-GELU expansion bias |
| `gpt_neox.layers.3.mlp.dense_h_to_4h.weight` | 512 × 128 | pre-GELU expansion projection |
| `gpt_neox.layers.3.post_attention_layernorm.bias` | 128 | LayerNorm offset |
| `gpt_neox.layers.3.post_attention_layernorm.weight` | 128 | LayerNorm scale |
| `gpt_neox.layers.4.attention.dense.bias` | 128 | attention output bias |
| `gpt_neox.layers.4.attention.dense.weight` | 128 × 128 | attention output projection |
| `gpt_neox.layers.4.attention.query_key_value.bias` | 384 | head-packed [head,Q/K/V,dimension] bias |
| `gpt_neox.layers.4.attention.query_key_value.weight` | 384 × 128 | head-packed [head,Q/K/V,dimension] projection |
| `gpt_neox.layers.4.input_layernorm.bias` | 128 | LayerNorm offset |
| `gpt_neox.layers.4.input_layernorm.weight` | 128 | LayerNorm scale |
| `gpt_neox.layers.4.mlp.dense_4h_to_h.bias` | 128 | post-GELU contraction bias |
| `gpt_neox.layers.4.mlp.dense_4h_to_h.weight` | 128 × 512 | post-GELU contraction projection |
| `gpt_neox.layers.4.mlp.dense_h_to_4h.bias` | 512 | pre-GELU expansion bias |
| `gpt_neox.layers.4.mlp.dense_h_to_4h.weight` | 512 × 128 | pre-GELU expansion projection |
| `gpt_neox.layers.4.post_attention_layernorm.bias` | 128 | LayerNorm offset |
| `gpt_neox.layers.4.post_attention_layernorm.weight` | 128 | LayerNorm scale |
| `gpt_neox.layers.5.attention.dense.bias` | 128 | attention output bias |
| `gpt_neox.layers.5.attention.dense.weight` | 128 × 128 | attention output projection |
| `gpt_neox.layers.5.attention.query_key_value.bias` | 384 | head-packed [head,Q/K/V,dimension] bias |
| `gpt_neox.layers.5.attention.query_key_value.weight` | 384 × 128 | head-packed [head,Q/K/V,dimension] projection |
| `gpt_neox.layers.5.input_layernorm.bias` | 128 | LayerNorm offset |
| `gpt_neox.layers.5.input_layernorm.weight` | 128 | LayerNorm scale |
| `gpt_neox.layers.5.mlp.dense_4h_to_h.bias` | 128 | post-GELU contraction bias |
| `gpt_neox.layers.5.mlp.dense_4h_to_h.weight` | 128 × 512 | post-GELU contraction projection |
| `gpt_neox.layers.5.mlp.dense_h_to_4h.bias` | 512 | pre-GELU expansion bias |
| `gpt_neox.layers.5.mlp.dense_h_to_4h.weight` | 512 × 128 | pre-GELU expansion projection |
| `gpt_neox.layers.5.post_attention_layernorm.bias` | 128 | LayerNorm offset |
| `gpt_neox.layers.5.post_attention_layernorm.weight` | 128 | LayerNorm scale |

## Static atlas protocol and limits

The runner is `minimind_re.transfer.atlas.write_atlas(directory, tensors, config)`. It accepts tensors only from an externally admitted loader and has no loading, downloading, tokenizing or forward capability. It creates a new output directory and refuses overwrite. Run under two CPU threads with deterministic algorithms set by the admitted caller. Target execution is **NOT RUN** as of this receipt.

1. Validate the complete key and shape map, CPU floating contiguous tensors and configuration. Compute every tensor's element count, finite/nonfinite counts, finite minimum/maximum/mean/L2/RMS in chunks of at most 65536 elements. Nonfinite original values stop all view/SVD work and remain visible in the report.
2. Select the fixed numeric vocabulary IDs 0..127 before any observations. For every block/head, export two 128×128 FP32 views: `E Q_h^T K_h E^T / sqrt(d_head)` and `E V_h^T O_h^T U^T`, using input embedding rows E and untied output rows U for those IDs. Output projection columns for head h form O_h. These 48 arrays require exactly 3,145,728 raw tensor bytes; NumPy headers and JSON add modest overhead. No full 50304×50304 matrix is constructed.
3. Compute singular values only, in FP64, for the first eight views in fixed layer/head/QK-then-OV traversal: layer 0, all four heads, QK and OV. Each matrix is at most 128×128. No randomized selection or fitted hypothesis is involved. CPU library/environment identity belongs in the surrounding admitted receipt; synthetic repetition checks equality within the current environment and does not guarantee cross-library bitwise equality.
4. Retain JSON statistics, sample IDs, limitations, singular values, safe `.npy` arrays and their SHA256 digests. These are static descriptors only. Raw full-tensor bytes need not be duplicated in the atlas because the original immutable manifest supplies tensor provenance.

QK views omit LayerNorm and biases and assume unrotated positions. Actual partial RoPE modifies only a subset of dimensions according to query/key relative positions, so the static QK view cannot determine real attention. OV views omit LayerNorm, attention weights, projection biases and contextual states; they describe a linear surrogate rather than an observed token transition or causal effect. Singular values of these sampled surrogates are not whole-model rank, semantic importance, feature discovery or intervention evidence.

Budget reservation proposal: zero model calls/token positions, <=120 CPU-process seconds, two threads, <=512 MiB atlas working RSS in addition to admitted source tensors, and <=16 MiB output. This is a conservative allocation for independent guard admission, not measured target performance. Abort on violated external time/memory/disk admission; the library itself does not implement a watchdog. Target payload finite statistics, sampled views, SVD, hashes and independent static audit remain **NOT RUN**.

Evidence sources: `evidence/transfer-002/implementation/header-map-v1.json`, `evidence/transfer-002/implementation/candidate-manifest-v1.json`, `evidence/transfer-002/selection-proposal.json`, and installed `PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py` (SHA256 `1a329e0f839999e74d485ea5d0cf6bae36f389577a51db4887d9ee7837a29ff8`).
