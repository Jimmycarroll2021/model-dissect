# GPT-2 source-assisted computation and exhaustive tensor map

Scope: `openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32 eager full-prefill inference. This document maps the inspected installed Transformers 5.1.0 implementation into the editable candidate. It is source-assisted reconstruction, not blind recovery. Target runtime parity requires separately recorded execution and independent verification; this map and synthetic checks do not establish it.

## Sources and exact exposure

Installed source hashes and current candidate provenance are recorded in `evidence/transfer-003/implementation/candidate-manifest-v3.json` (manifest SHA256 `d768bf9ccabc67a864baa14e3f0fd402c374ad3bb1f76ff9e89a0d67310b7f6e`; candidate source SHA256 `e1f8762cfd16bfcf8263c62a0cbd28ef4305ea1eb0463e820091c693357f6c96`). Candidate v3 adds the observation-only hook; the tensor map and computation equations below are unchanged. The installed sources used here are:

- GPT-2 modeling (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:53`; external or omitted historical artifact): eager attention, lines 53–94; boolean causal buffer, lines 100–110; packed Q/K/V splitting and reshape, lines 229–241; flatten and output projection, lines 265–266; residual block, lines 317–361; embedding addition, lines 645–657; final normalization, line 724; tying declaration, line 755; output logits, line 823.
- Conv1D (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/pytorch_utils.py:97`; external or omitted historical artifact): stored matrix is input-by-output; line 121 uses `addmm(bias, flattened_input, weight)`.
- GELU-new (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/activations.py:59`; external or omitted historical artifact): line 66 fixes multiplication order and constants.
- Configuration (`PRIVATE_HOME/.local/lib/python3.12/site-packages/transformers/models/gpt2/configuration_gpt2.py:132`; external or omitted historical artifact): GPT-2 dimension and behavioral defaults.

The coordinator supplied actual checkpoint header identity `0c58f57dc9d870378cc1195c914ce7065e45e6eacc79fadfc5ef33dd32084181`, reporting 160 F32 tensors: 148 learned parameter tensors and 12 serialized causal buffers. The base checkpoint uses `wte.weight`, `wpe.weight`, and `h.N.*`; native LM-head model names prepend `transformer.`. The candidate accepts either complete namespace and rejects mixed names. No target tensor loading was performed by this atlas implementation assignment.

## Exhaustive tensor partition

Let width D=768, heads H=12, per-head width d=64, vocabulary V=50,257, context P=1,024, MLP width I=3,072, and layers L=12. Every checkpoint tensor belongs to one row below; `{l}` ranges independently over **all integers 0 through 11**.

| Checkpoint key | Shape | Count | Role |
|---|---|---:|---|
| `wte.weight` | V × D | 1 | Learned token embedding E; also tied unembedding |
| `wpe.weight` | P × D | 1 | Learned absolute position embedding A |
| `h.{l}.ln_1.weight`, `h.{l}.ln_1.bias` | D each | 24 | First normalization scale and offset |
| `h.{l}.attn.c_attn.weight` | D × 3D | 12 | Packed input-to-Q, input-to-K, input-to-V columns |
| `h.{l}.attn.c_attn.bias` | 3D | 12 | Packed Q/K/V offsets |
| `h.{l}.attn.c_proj.weight` | D × D | 12 | Packed head-output-to-residual projection |
| `h.{l}.attn.c_proj.bias` | D | 12 | Attention output offset |
| `h.{l}.ln_2.weight`, `h.{l}.ln_2.bias` | D each | 24 | Second normalization scale and offset |
| `h.{l}.mlp.c_fc.weight` | D × I | 12 | MLP expansion |
| `h.{l}.mlp.c_fc.bias` | I | 12 | MLP expansion offset |
| `h.{l}.mlp.c_proj.weight` | I × D | 12 | MLP contraction |
| `h.{l}.mlp.c_proj.bias` | D | 12 | MLP contraction offset |
| `ln_f.weight`, `ln_f.bias` | D each | 2 | Final normalization scale and offset |
| `h.{l}.attn.bias` | 1 × 1 × P × P | 12 | Serialized lower-triangle causal mask; **not learned** |
| `lm_head.weight` | V × D | 0 stored in this checkpoint | Logical alias of E; never counted twice |

Learned total: **148 tensors, 124,439,808 elements**. Causal buffers: **12 tensors, 12,582,912 elements**. Stored total: **160 tensors, 137,022,720 elements**. The atlas computes counts from admitted tensors, rather than substituting these expected values. An optional historical `attn.masked_bias=-10000` is classified separately if present; it is not a learned parameter or a substitute for modern finite-min masking.

## Forward equations and observable boundaries

All vectors multiply matrices from the left. For token IDs t and explicit/default absolute positions p:

`X₀ = E[t] + A[p]` → **embedding** boundary. Dropout is identity in evaluation.

`LN(X; γ,β) = (X - mean(X)) / sqrt(var(X, unbiased=False) + ε) * γ + β`, with ε from configuration (target default 1e-5), computed over the final D axis using PyTorch layer normalization.

For each layer l:

1. `U = LN₁(X)`; `[Qpacked | Kpacked | Vpacked] = U Wc_attn + bc_attn`. Split contiguous D-wide sections, then each into H contiguous d-wide heads. Keep the single packed addmm operation; do not replace it with separately evaluated head projections.
2. For each head: `S = Q Kᵀ / sqrt(d)`; replace positions above the causal diagonal by float32 finite minimum. Supported binary key masks add the same finite-min value to hidden keys. `Pattn = softmax(S, final_axis)` and `Z = Pattn V`.
3. Transpose into `[B,Q,H,d]`. This is the **head intervention seam**; native pre-hook on `attn.c_proj` observes the flattened `[B,Q,D]` equivalent. Apply any per-call transform here, then flatten and evaluate the original packed output `Zpacked Wc_proj + bc_proj`.
4. `R = X + attention_output`.
5. `M = LN₂(R) Wfc + bfc`; `GELUnew(M) = 0.5 M (1 + tanh(sqrt(2/π) (M + 0.044715 M³)))`.
6. `Xnext = R + GELUnew(M) Wmlp_proj + bmlp_proj` → **block{l}** boundary, after optional block-output transform.

`F = LNfinal(X₁₂)` → **final_norm**; `logits = F Eᵀ` → **logits**, covering every vocabulary entry at every supplied query position. No output bias exists.

The supported runtime subset rejects cache inputs, left padding, training, cross-attention, pruned heads, autocast, alternate attention implementations and alternate behavioral flags. Right padding is supported with a visible first key; scientific next-token metrics must select the last real query position. Learned position IDs remain explicit and need not be monotonically increasing.

## Static atlas and interpretation limits

`build_atlas(config, tensors, sample_heads=None, chunk_elements=262144, svd_count=0)` consumes externally admitted CPU float32 tensors. It performs no loading, downloading, tokenization, model forward or generation. It returns `(metadata, raw_matrices)`; serialization and output quota enforcement belong to the runtime owner.

Every stored tensor receives shape, role, finite check, min/max, mean, RMS and L2 statistics using float64 chunk accumulations. Causal buffers receive their own labeled rows, full finite checks and exact rowwise lower-triangle comparison; they are excluded from learned-parameter totals. Tied aliases are separately labeled and equality checked. All 144 heads receive separate Q/K/V/O Frobenius norms.

For head h, columns `h*d:(h+1)*d` of each packed Q/K/V section define `WQ`, `WK`, `WV`; corresponding rows of the packed output matrix define `WO`. The raw static matrices are:

- **QK = WQ WKᵀ**, D × D, unscaled. For row vectors u and v, `u QK vᵀ` is the bias-free unscaled bilinear score contribution.
- **OV = WV WO**, D × D. For row vector u, `u OV` is its bias-free value/output transform before attention-weighted context mixing.

These omit layer normalization, projection biases, learned position contributions, softmax and context; they do not identify semantics or establish causal effects. Attention projection biases are included in actual candidate forward, not silently absorbed into these matrices.

Default sampled sites are `(layer i, head i)` for i=0..11 in this target. These are predefined atlas samples, not science-selected important heads. If the separately frozen science sites differ, preserve and disclose the difference. At most 12 sampled sites produce 24 float32 D × D raw matrices: **56,623,104 bytes** (54 MiB). Up to four optional SVDs return singular values only, selected by recorded lexicographic matrix-key order; default is zero. No token projections or vocabulary-square matrices are created.

For target resource admission, retained matrices require 54 MiB; chunk statistics allocate approximately 4–12 MiB of scratch plus temporary contiguous head slices and matrix multiplication workspace. Reserve **at least 128 MiB additional working memory beyond already-loaded input tensors and Python/PyTorch overhead** with SVD disabled. This is a conservative allocation estimate, not a measured target RSS ceiling. SVD runtime/workspace is uncalibrated and requires separate admission. Static packet consumes **zero model forward positions**. CPU threads and wall time must be externally bounded. Target atlas values and exact 12-buffer validation remain unexecuted until separately admitted.
