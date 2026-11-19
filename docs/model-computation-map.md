# How the MiniMind weights become a prediction

This guide maps **all 91 stored tensors (68,827,392 values)** to the computation they participate in. It answers what each array does mathematically. It does not assign a learned meaning to each number, claim that a head recognizes a particular concept, or establish which pathways explain a useful behavior. Those require observations and controlled interventions on real inputs.

The companion [tensor map](../evidence/computation-map/tensor-map.json) has one entry per exact checkpoint key, including shape, storage group, equation, input/output dimensions, atlas row, and hashed source with line ranges. [The manifest](../evidence/computation-map/manifest.json) binds this guide and map to their source evidence. This task read existing JSON and source files; it did not load the checkpoint, tokenize text, run inference, recompute spectra or modify weights.

The artifact is `full_sft_768.pth`, checkpoint SHA-256 `5b8c49f6c9d965092e651cafeeaeb8705558632b3fd4ac8ab319cbc5a3cbc4a0`, as recorded in the existing inventory. Identity was reused from that receipt, not freshly recomputed over checkpoint bytes. Dimensions and storage come from [the inventory](../artifacts/allowed/tensor_inventory.json); values/statistics remain in the accepted atlas (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/runs/atlas-v1/atlas.json`; external or omitted historical artifact). Operation meanings are **source-assisted** from the [pinned vendor source](../oracle/vendor/model_minimind.py) and [editable primitive reconstruction](../src/minimind_re/forward.py), with [supplied constants](../artifacts/allowed/supplied_config.json). This is not discovery of a unique architecture from unnamed weights.

## The complete accounting

There are eight transformer blocks, numbered 0 through 7. Each contains eleven stored tensors. One embedding, one final normalization vector and one output matrix supply the other three: **8 × 11 + 3 = 91**.

| Exact key or suffix | Shape | Values per tensor | What it contributes |
|---|---:|---:|---|
| `model.embed_tokens.weight` | 6400 × 768 | 4,915,200 | A 768-number starting vector for each token ID |
| `model.layers.L.input_layernorm.weight` | 768 | 768 | Attention-input normalization scale |
| `model.layers.L.self_attn.q_proj.weight` | 768 × 768 | 589,824 | Eight query vectors, each 96 wide |
| `model.layers.L.self_attn.k_proj.weight` | 384 × 768 | 294,912 | Four key vectors, each 96 wide |
| `model.layers.L.self_attn.v_proj.weight` | 384 × 768 | 294,912 | Four value vectors, each 96 wide |
| `model.layers.L.self_attn.q_norm.weight` | 96 | 96 | One scale vector shared by the eight query heads |
| `model.layers.L.self_attn.k_norm.weight` | 96 | 96 | One scale vector shared by the four key heads |
| `model.layers.L.self_attn.o_proj.weight` | 768 × 768 | 589,824 | Returns concatenated attention outputs to residual coordinates |
| `model.layers.L.post_attention_layernorm.weight` | 768 | 768 | MLP-input normalization scale |
| `model.layers.L.mlp.gate_proj.weight` | 2432 × 768 | 1,867,776 | Nonlinear gate branch |
| `model.layers.L.mlp.up_proj.weight` | 2432 × 768 | 1,867,776 | Parallel value branch |
| `model.layers.L.mlp.down_proj.weight` | 768 × 2432 | 1,867,776 | Returns gated MLP activity to residual coordinates |
| `model.norm.weight` | 768 | 768 | Final normalization scale |
| `lm_head.weight` | 6400 × 768 | 4,915,200 | Converts final representation into vocabulary logits |

`L` means every layer from 0 to 7, without exceptions. Each block contributes 7,374,528 values. The total is `8 × 7,374,528 + 2 × 4,915,200 + 768 = 68,827,392`. All inventory entries are FP16; the reconstruction explicitly converts them to FP32 for its CPU computation. These are stored-value counts, not proof of that many independently learned parameters. No bias arrays, learned positional embeddings or MoE expert arrays occur in this checkpoint inventory. RoPE sine/cosine buffers in the source are nonpersistent, computed buffers rather than additional stored learned tensors.

**Embedding and head tying needs a distinction.** The checkpoint inventory gives embedding and head different storage groups, 0 and 90, and the reconstruction reads them as separate arrays: it does not enforce weight tying. That is an untied representation in the reconstructed computation. However, the vendor constructor defaults `tie_word_embeddings=True` and assigns the embedding parameter to the output parameter (source lines 30 and 241–247); the reference adapter uses that default. Separate saved storage does not establish independently trained values or numerical inequality. This guide does not re-test their full numerical equality. Preserve both exact keys rather than silently replacing either matrix or counting historical training degrees of freedom from storage alone.

## Equations, with actual dimensions

Use row vectors. A stored matrix has shape `[output,input]`, so its linear operation is `x Wᵀ`. `B` is batch size and `T` is the current full-prefix length. The admitted reconstruction interface supports CPU arrays and its historical evaluation uses batch one. The explanation below uses **full-prefill, cache off**, dropout zero, explicit attention matmul and softmax, unscaled split-half RoPE. The source also contains other paths; this document does not establish their numerical equivalence.

### 1. Lookup and normalization

For token IDs `ids[B,T]`, lookup produces:

```text
x₀[b,t,:] = E[ids[b,t],:]                         [B,T,768]
RMS(z,γ) = z ⊙ γ / sqrt(mean_last(z²) + 10⁻⁶)
```

RMS divides by the root mean square along the final axis, then applies the learned coordinate scales. It does **not** subtract a mean. For a 768-vector the mean contains 768 squared values; for a head vector it contains 96. The epsilon is supplied/source-derived, not recoverable uniquely from tensor shapes. The vendor RMS implementation computes normalization in FP32 and casts back; this reconstruction already uses FP32 weights and activations.

### 2. Each block prepares queries, keys and values

For each `l = 0,…,7`:

```text
n = RMS(x_l, γ_in_l)                              [B,T,768]
q₀ = reshape(n Wqᵀ)                              [B,T,8,96]
k₀ = reshape(n Wkᵀ)                              [B,T,4,96]
v  = reshape(n Wvᵀ)                              [B,T,4,96]
q₁ = RMS(q₀, γ_q_l)                              [B,T,8,96]
k₁ = RMS(k₀, γ_k_l)                              [B,T,4,96]
```

The same 96 query scales apply separately to all eight query heads; the same 96 key scales apply separately to all four key heads. Values receive neither this head RMS normalization nor RoPE. These steps are at vendor lines 100–119 and reconstruction lines 36–38.

### 3. Position rotates normalized queries and keys

For position `p`, coordinate-pair index `j = 0,…,47`, and either normalized head vector `z`:

```text
ω_j = 1,000,000^(−2j/96)
α = p ω_j
RoPE(z,p)[j]    = z[j] cos(α) − z[j+48] sin(α)
RoPE(z,p)[j+48] = z[j+48] cos(α) + z[j] sin(α)
q = RoPE(q₁,p); k = RoPE(k₁,p)
```

This pairs the first half of the 96 coordinates with the second half, not adjacent even/odd coordinates. Position changes the query/key geometry; it is not a learned vector added to the embedding. The reconstruction accepts explicit `position_ids[B,T]`. The vendor model's outer forward normally slices consecutive positions from cached length; the reference adapter explicitly selects the supplied positions. Thus position-interface handling matters when comparing padded or cached runs. Source lines 62–89 and 214–224, reference adapter lines 29–34, and reconstruction lines 19–21 document that distinction. Configured maximum position is 32768; the equation alone does not prove useful performance at that length.

### 4. Grouped attention selects and mixes earlier values

There are eight query heads but only four key/value heads. `repeatKV` repeats each KV head twice, in this order:

| Query head `h` | Shared KV head `floor(h/2)` |
|---|---|
| 0, 1 | 0 |
| 2, 3 | 1 |
| 4, 5 | 2 |
| 6, 7 | 3 |

For query position `t`, key position `s`, and head `h`, omitting batch index:

```text
score[h,t,s] = dot(q[t,h,:], k[s,floor(h/2),:]) / sqrt(96)
score[h,t,s] = −∞                                 when s > t
score[h,t,s] += −10⁹                              when key s is padding
A[h,t,:] = softmax(score[h,t,:])                   over key positions s
c[t,h,:] = Σ_s A[h,t,s] v[s,floor(h/2),:]           96 values
c_concat[t,:] = concatenate(c[t,0,:],…,c[t,7,:])    768 values
a[t,:] = c_concat[t,:] Woᵀ                        768 values
y = x_l + a                                      [B,T,768]
```

Scores and attention probabilities have shape `[B,8,T,T]`; per-head contributions have shape `[B,T,8,96]`. Causal masking excludes future keys. Padding receives the implementation's finite additive `−10⁹`; it is not literally a mathematical `−∞` mask. Padding masks keys, not query rows, so a padded query can still have an output. The reconstruction rejects inputs with no unmasked key anywhere in a batch row and checks final logits are finite; this is not a claim that every conceivable mask arrangement produces meaningful rows. Softmax normalizes over keys, not heads or residual coordinates. Dropout is zero for the documented configuration.

A head's 96 outputs occupy columns `96h:96(h+1)` in `Wo`. Consequently its projected contribution is `c_h Wo[:,96h:96(h+1)]ᵀ`. Adding these eight projected contributions yields `a`. This algebra specifies where to measure a head's contribution; it supplies no evidence that one head actually dominates on a particular prompt. See vendor lines 124–134 and reconstruction lines 40–46.

For cached execution, the source concatenates **already rotated keys** and unrotated values from previous positions before KV repetition. With `P` previous positions, cache tensors per layer have shape `[B,P,4,96]`, scores become `[B,8,T,P+T]`, and a query at new index `t` may attend through key index `P+t`. No new learned tensor appears. These are static code facts only: the project's historical cache/full fidelity discrepancy remains unresolved and FAIL.

### 5. The gated MLP supplies another residual update

```text
u = RMS(y, γ_post_l)                              [B,T,768]
g_pre = u Wgateᵀ                                 [B,T,2432]
b = u Wupᵀ                                       [B,T,2432]
SiLU(z) = z / (1 + exp(−z))
h = SiLU(g_pre) ⊙ b                              [B,T,2432]
d = h Wdownᵀ                                     [B,T,768]
x_(l+1) = y + d                                  [B,T,768]
```

This is a SiLU-gated MLP, commonly called SwiGLU. The two 2432-wide branches are multiplied coordinate by coordinate. The gate is not a Boolean switch and SiLU is not bounded to `[0,1]`. The down projection combines all gated coordinates into the residual stream. No single fixed linear matrix captures this nonlinear, input-dependent path. Each block adds two updates, attention then MLP, while keeping the residual width at 768. See vendor lines 136–146 and 191–199; reconstruction lines 46–53.

### 6. Final representation becomes next-token scores

```text
f = RMS(x₈, γ_final)                              [B,T,768]
logits = f Wlmᵀ                                  [B,T,6400]
P(next_token=k | prefix ending at t)
    = exp(logits[t,k]) / Σ_j exp(logits[t,j])
```

The model forward returns logits. The final softmax above describes the unmodified next-token distribution; decoding rules can subsequently choose argmax or apply temperature, sampling or other changes. Neither a word string nor a full response is a direct matrix output. Tokenization determines input IDs and decoding maps output IDs to text; those procedures are outside the 91-tensor computation and are not exercised here. Source lines 235 and 250–253 and reconstruction line 55 locate the final operations.

## One token's schematic journey

Consider an unspecified real token ID at position `t` in an unspecified prefix. This is a schematic, not a claimed observed prompt or recorded activation trace.

1. Its ID selects one row from the 6400-row embedding table, giving 768 numbers. We have not selected an actual token or read its row values in this task.
2. In block 0, RMS normalization and projections produce eight normalized, rotated queries, four normalized, rotated keys and four values. Its queries score the allowed earlier positions and itself. The prefix matters: the same token ID can obtain different attention results in different contexts.
3. Eight weighted value mixtures are projected back to 768 coordinates and added to the token's current state. The MLP reads the normalized updated state, forms 2432 gated products, projects them down and adds its update.
4. The token passes through blocks 1–7 in the same structural order. Later blocks read representations already modified by earlier blocks, including context mixed through attention.
5. Final normalization and the output matrix produce 6400 logits. At the last prefix position, these score the next token. Appending a selected token and evaluating again is a separate generation step.

**Unobserved here:** actual input IDs, embedding values for the example, normalization denominators, Q/K/V activations, scores, attention probabilities, head outputs, MLP gates/products, both residual updates, final vectors and logits. The existing atlas supplies static weight statistics; it does not fill in those input-dependent quantities. Other project experiments retained selected observations, but this guide does not turn them into a complete trace of this schematic token.

## What this lets us inspect, and what remains unknown

The map makes every saved tensor addressable by its operation, dimensions and upstream/downstream quantities. It connects the atlas's stored-value views to an editable computation. For example, a raw QK product omits head RMS normalization and positional rotation; a raw OV composition omits input normalization and attention weighting. Those products are useful static views with narrower meanings than the complete running pathway.

No semantic feature label is inferred from a large weight, singular direction, token cosine similarity, gate coordinate or head index. The [prior controlled investigation](weight-dissection-result.md) rejected its frozen static-OV prediction; it did not establish a useful-behavior mechanism. The [broader goal](reverse-engineering-goal.md) still requires input-dependent explanation and controlled causal evidence.

Historical same-mode reference fidelity is narrower than universal equivalence, and this task did not rerun it. The original cache/full discrepancy remains FAIL. This map provides complete **key-to-computation accounting**, not a proof that all internal meanings or all model behaviors are understood. Independent review of the exact guide/map candidate is a separate acceptance step.
