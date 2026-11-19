# An actual saved input through MiniMind

This walkthrough follows the **last token, position 10**, through all eight layers of a saved 11-token CPU FP32 full-prefill execution. It connects actual captured arrays to the [complete 91-tensor computation map](model-computation-map.md). The [self-contained viewer](../visualization/actual-input/index.html) shows all eight attention heatmaps and numerical tables; the [trace summary](../evidence/actual-input/trace-summary.json) retains exact scalar values and provenance.

**This is an artificial diagnostic fixture**, `continuation-length-8-none`, extended with its first three prescribed continuation IDs. They were supplied by the fixture, not generated for this walkthrough. There is no natural-chat interpretation or useful-behavior claim. The analysis loads only saved capture archives with `weights_only=True` on CPU, reads JSON, and derives scalar summaries. It does not load checkpoint weights, run inference, execute tokenization, or recompute QK/attention/value contractions.

## Exactly what went in

| Position | Token ID | Individually decoded byte piece |
|---|---:|---|
| 0 | 92 | `"Y"` |
| 1 | 611 | `"est"` |
| 2 | 311 | `"er"` |
| 3 | 1753 | `"day"` |
| 4 | 47 | `","` |
| 5 | 4920 | `" several"` |
| 6 | 6250 | `" music"` |
| 7 | 108 | `"i"` |
| 8 | 869 | `" M"` |
| 9 | 104 | `"e"` |
| 10 | 325 | `"an"` |

Positions 0–7 are the original eight-token fixture; positions 8–10 are the prescribed extension. All eleven attention-mask entries are true and position IDs are 0–10. The final query therefore has eleven legal key positions, including itself.

Pieces above are reconstructed by looking up the pinned `tokenizer.json` vocabulary and reversing its ByteLevel byte alphabet, then decoding each byte piece as UTF-8. This is a transparent standard-library decoding operation, not a tokenizer invocation or a claim about word boundaries. Raw pieces and byte hex are retained in the JSON; for other token IDs a replacement character can indicate an incomplete byte sequence. Token spellings identify the measured input, not semantic features.

## From embedding through eight blocks

Token 325 (`"an"`) starts with a saved 768-coordinate embedding whose L2 norm is **1.34620758** and RMS is **0.04857708**. It exactly matches the captured layer-0 input at this position. The full-prefix embedding capture has shape `[1,11,768]`.

Every layer has saved normalized/rotated queries `[1,8,11,96]`, keys and values `[1,4,11,96]`, unprojected attention `[1,11,768]`, projected attention and residual states. Query head `h` shares key/value head `floor(h/2)`. These dimensions and operations are explained in the computation map. The Q/K captures are after head normalization and positional rotation; they do not expose the preceding normalization denominators.

The table summarizes the **same last-token position** at each layer. Norm means Euclidean L2 norm. Attention net change is `post_attention − pre`; MLP net change is `post_block − post_attention`, subtracted in float64 from the saved FP32 states. Because the original residual additions rounded, these differences are **net residual changes, not exact pre-addition branch activations**. Direct projected-attention norms are also retained separately in the JSON.

| Layer | Before block | Attention net change | After attention | MLP net change | After block |
|---|---:|---:|---:|---:|---:|
| 0 | 1.34621 | 3.20303 | 3.37903 | 4.31024 | 5.39789 |
| 1 | 5.39789 | 4.09062 | 5.00315 | 4.88922 | 6.51057 |
| 2 | 6.51057 | 4.09727 | 6.78131 | 6.36826 | 8.62625 |
| 3 | 8.62625 | 5.45983 | 9.96755 | 6.38745 | 9.53464 |
| 4 | 9.53464 | 10.30762 | 14.23085 | 9.13161 | 14.14099 |
| 5 | 14.14099 | 14.81024 | 21.70544 | 12.62590 | 19.11353 |
| 6 | 19.11353 | 16.16755 | 30.62923 | 21.46098 | 26.60207 |
| 7 | 26.60207 | 15.82281 | 36.55317 | 30.11186 | 32.97941 |

Each saved post-block vector exactly equals the next layer’s pre-block vector. This checks that the displayed chain follows the same captured execution. A larger norm does not establish greater task importance: contributions have directions, can reinforce or cancel, and can be transformed by downstream operations. The table therefore does not rank causal influence.

## Which positions receive attention

The viewer has one 8-head × 11-position heatmap per layer, all using the same probability scale. The values come from the diagnostic’s **saved full-shape primitive replay** on captured native Q/K/V. They were not directly hooked at the native softmax line and were not recomputed in this task. For every layer, the replay’s saved weighted-value vector exactly matches the last-token unprojected attention captured inside the native forward. That connects the saved probability view to the corresponding attention output without claiming it is a separate direct softmax capture.

A compact view below gives the peak key position for each query head, in head order 0–7. Full-precision probability rows, entropy in nats, per-head query norms and weighted-value norms are in the JSON. The viewer shows rounded probability rows, peaks and entropy. Every row sums to one within `1e-6`.

| Layer | Peak key positions for heads 0–7 | Peak probabilities for heads 0–7 |
|---|---|---|
| 0 | 10, 9, 10, 10, 10, 10, 9, 9 | 0.699, 0.650, 0.804, 0.743, 0.699, 0.826, 0.439, 0.472 |
| 1 | 10, 9, 10, 9, 9, 10, 9, 9 | 0.372, 0.449, 0.524, 0.626, 0.423, 0.533, 0.642, 0.532 |
| 2 | 10, 8, 9, 10, 10, 10, 8, 10 | 0.613, 0.366, 0.413, 0.542, 0.349, 0.539, 0.274, 0.322 |
| 3 | 10, 4, 10, 10, 10, 9, 4, 4 | 0.341, 0.413, 0.435, 0.432, 0.263, 0.414, 0.357, 0.413 |
| 4 | 10, 10, 8, 10, 4, 8, 10, 10 | 0.608, 0.506, 0.275, 0.318, 0.312, 0.238, 0.583, 0.359 |
| 5 | 8, 10, 6, 6, 5, 8, 10, 10 | 0.466, 0.274, 0.347, 0.384, 0.343, 0.228, 0.585, 0.367 |
| 6 | 10, 10, 10, 10, 6, 8, 8, 10 | 0.251, 0.250, 0.424, 0.555, 0.378, 0.347, 0.250, 0.609 |
| 7 | 6, 10, 4, 6, 10, 10, 10, 10 | 0.377, 0.543, 0.745, 0.554, 0.365, 0.926, 0.721, 0.711 |

A peak tells us where a particular attention distribution places its largest mass for this input. It does not tell us that the head recognizes the associated token string, that this position causes the output, or that removing the head would have a predictable effect. The weighted values and output projection also matter. The [separate ECHO intervention result](echo-mechanism-result.md) concerns its own fixtures and controls; its causal finding must not be attributed to this different diagnostic prefix.

## Final normalization and next-token scores

After block 7 the last-token residual norm is **32.97941316**. The captured final normalized representation has norm **51.24870730** and RMS **1.84927843**. It is not unit-length normalization: learned coordinate scales follow RMS normalization. The saved final output contains 6400 logits at each of eleven positions.

For the last position, the leading scores are below. Probabilities are newly derived offline from the saved logits using stable float64 scalar `exp(logit − max_logit) / sum_exp` over all 6400 entries. They are an unmodified distribution, with no temperature, sampling or generation.

| Rank | Token ID | Decoded byte piece | Saved logit | Derived probability |
|---|---:|---|---:|---:|
| 1 | 869 | `" M"` | 14.01930141 | 41.873395% |
| 2 | 325 | `"an"` | 13.79243088 | 33.374099% |
| 3 | 80 | `"M"` | 11.49971581 | 3.370513% |
| 4 | 127 | `"|"` | 11.42984390 | 3.143048% |
| 5 | 100 | `"a"` | 11.23093128 | 2.576110% |
| 6 | 1817 | `" |"` | 10.71110821 | 1.531821% |
| 7 | 1156 | `"ary"` | 10.50637150 | 1.248223% |
| 8 | 256 | `" "` | 10.24563217 | 0.961733% |
| 9 | 5257 | `"man"` | 10.13828850 | 0.863845% |
| 10 | 114 | `"o"` | 9.79723930 | 0.614214% |

This establishes the measured next-token preference at this one prefix. It does not establish that a complete continuation would be correct, coherent or useful, and no token was appended during this analysis.

## Evidence and remaining gaps

The two capture-file hashes were checked against the original producer result before reading them. All native layer tensors in `native-full.pt` matched their duplicate entries in `all-operands.pt` exactly. The summary records these checks, residual-chain equality and the eight saved primitive/output matches. The [manifest](../evidence/actual-input/manifest.json) binds the guide, viewer, generator, numeric summary and their source files. The existing diagnostic independent repeat is upstream evidence; this new offline interpretation still requires its own independent review.

Directly captured: embedding, per-layer Q/K/V, projected and unprojected attention, residual states, final representation and logits. Saved diagnostic replay: last-query scores, probabilities and weighted values. Newly derived offline: byte-piece displays, L2/RMS summaries, residual differences, attention row summaries and next-token probabilities.

Still unavailable here: exact RMS denominators, pre-normalization head arrays, MLP gate/up/product arrays and exact pre-addition MLP output. No new causal experiment or semantic feature identification is supplied. The selected last-token trace is an input-dependent observation across all layers, not an account of every activation for every input. Historical cache/full limitations remain outside this full-prefill account; it makes no new cache parity claim.
