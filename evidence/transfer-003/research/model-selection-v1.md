# Third-model selection v1

Recommendation: **openai-community/gpt2 (GPT-2 small, advertised 124M)**, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. This is research and an acquisition proposal, not runtime admission or evidence of model fidelity. No checkpoint downloaded, dependency installed, or model run. Original `EXTERNAL_ARTIFACT_ROOT` artifacts were untouched.

## Comparison (live upstream metadata)

| Candidate | Immutable revision | Checkpoint bytes | Selected metadata/tokenizer bundle included, total bytes | Architecture and decision |
|---|---|---:|---:|---|
| GPT-2 small | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 548105171 | 550967953 | 12 layers, hidden768, 12 heads, context1024, vocab50257; learned absolute positions, LayerNorm, GELU-new, sequential residual, Conv1D projections and tied readout. Recommended for new portability coverage. |
| Pythia70M | `a39f36b100fe8a5377810d56c3f4789b9c53ac42` | 166029852 | 168158880 | 6 layers, hidden512, 8 heads, context2048, vocab50304; parallel residual, partial RoPE, untied readout. Cheapest reuse but largely repeats the second architecture. |
| SmolLM2-135M | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 269060552 | 272443805 | 30 layers, hidden576, 9 query/3 KV heads, context8192, vocab49152, RMSNorm/SiLU/RoPE, tied readout. Modern Llama tooling but repeats more MiniMind motifs and deeper boundary matrix. |

Sizes are exact remote API metadata, not locally verified bytes. Safetensors file size is not a parameter-count proof; inspect every tensor, buffer and alias after acquisition. Pythia's historical naming and SmolLM2's paper should not replace exact artifact inspection.

GPT-2 checkpoint SHA256 advertised by upstream LFS: `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`. The JSON records exact filenames/sizes and each compared config. GPT-2 selected bundle includes safetensors, config, generation config, tokenizer JSON/config, vocabulary/merges, and model card. Pin every request and permit no fallback to main. Model-card license is MIT; Pythia and SmolLM2 cards specify Apache-2.0. GPT-2 original code is MIT. Preserve model card/license attribution. GPT-2 is open weights/code; do not imply its full WebText corpus was released.

[GPT-2 pinned config](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json), [card/license](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/README.md), [tokenizer](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json), [Pythia config](https://huggingface.co/EleutherAI/pythia-70m/blob/a39f36b100fe8a5377810d56c3f4789b9c53ac42/config.json), [SmolLM2 config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json).

Tokenizer: GPT-2 byte-level vocabulary/merge tokenizer; 50257 entries, end-of-text ID50256 used as BOS/EOS in config. Observed tokenizer JSON has ByteLevel pre-tokenization, add_prefix_space=false, null normalizer and one end-of-text special token. Freeze IDs and explicit masks; do not assume a native padding token or borrow another tokenizer. Installed loader compatibility remains to be checked.

## Current technology and paper-driven changes

The live [TransformerLens README](https://github.com/TransformerLensOrg/TransformerLens/blob/284e35b4ee2deea0924894c221a3be8ad519474d/README.md) recommends TransformerBridge and states v4 removed legacy HookedTransformer.from_pretrained. It claims default raw-HF weight preservation; compatibility mode changes numerics. Therefore old examples and automatic folding/centering are unsafe assumptions for this task. This is an upstream claim, not local verification. Keep existing native Transformers oracle plus independently composed tensor implementation as the numerical foundation; consider TransformerBridge only as a separately verified optional inspection adapter.

Current [NNsight README](https://github.com/ndif-team/nnsight/blob/ff4c2a3518e95cfdcd8d0745599be40784e8e2ca/README.md) directly demonstrates GPT-2 TransformersModel tracing and block-output intervention. It is a credible optional native intervention workflow, but installation/API compatibility must be tested separately. [Pyvene](https://github.com/stanfordnlp/pyvene/tree/9e333904dcf9e597ca76170010d17f4d4580de8d) and its [paper](https://arxiv.org/abs/2403.07809) offer explicit intervention specifications; adopt auditable site/position/source/target records without requiring another dependency. No framework is itself an independent mathematical reconstruction.

[GPT-2's primary report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) motivates the distinct architecture and language-model behavior. [Pythia's paper](https://arxiv.org/abs/2304.01373) supports controlled fixed-identity comparisons. [SmolLM2's paper](https://arxiv.org/abs/2502.02737) primarily describes 1.7B training; do not transfer its headline scores to the135M checkpoint.

[Zhang/Nanda](https://arxiv.org/abs/2309.16042) show patching conclusions depend on corruption and metric choices; [Heimersheim/Nanda](https://arxiv.org/abs/2404.15255) discuss interpretation limits. Accordingly freeze contrast/corruption, logit-difference direction, eligibility denominator, development-only selection, random/matched controls, no-op/restoration and held-out criteria before observations. Report a bounded causal effect of the selected intervention, never universal semantic understanding. Do not select a promising layer from held-out outcomes.

## Lessons retained from second-model blockers

Declare full-prefill source-assisted reconstruction as the primary numerical scope before running. Preserve strict cached/full mismatch separately; do not spend an open-ended repair loop on shape-dependent floating-point kernels or silently replace the native anchor. A tiny diagnostic compares full-prefix native/candidate outputs first; any cache experiment has its own preregistered gate and receipt. Native readout, activation function, projection orientation, padding/position semantics and tied aliases must be inspected explicitly. Same-mode numerical success is not cross-mode success. Earlier Pythia failure remains failed, irrespective of third-model results.

Build a minimal architecture map, checkpoint inventory, independent forward, per-block and final-logit parity, actual-input trace, then one controlled investigation and an independent execution repeat. Preserve failures and charge every call. The third result completes the user's three-model total only after its own scoped criteria pass; MiniMind and Pythia receipts must retain their exact existing scope.

## Resource estimate and next admission

Architecture arithmetic gives124439808 unique trainable parameters and497759232 FP32 parameter bytes; verify against acquired tensors. One native plus candidate copy is roughly0.93GiB parameters before framework/activation/raw evidence overhead. Estimated short-prompt peak RAM2–6GiB; propose10GiB cap and two CPU threads with serialized processes. GPU is unnecessary for the proposed bounded task; no free-resource or GPU availability claim is made. Reserve600MiB acquisition, separately budget evidence disk from exact frozen shapes. Avoid storing redundant full-vocabulary arrays for every token unless required by acceptance; never omit dimensions from an accepted comparator.

No honest full-run wall-time estimate is available without a pilot. Admit a <=120-second short-prompt pilot first, then estimate the unchanged verification schedule from measured calls/positions and reserve independent repetition capacity. Do not automatically inherit Pythia's250000-position or timing ceiling. No training, fine-tuning or generation benchmark is needed. Verify current CPU/memory/disk ownership and installed Transformers behavior before loading. Download preservation, independent identity verification and runtime guard approval remain next work, not research accomplishments.

Memory was used only to route existing source-assisted/oracle separation guidance (MEMORY.md71–72); current configuration and technology claims above were checked against live primary sources.
