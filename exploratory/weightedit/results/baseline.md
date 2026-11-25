# Baseline result (spec 005, Story 1)

Run 2026-11-25 under `specs/005-weight-edit-coding/baseline-protocol.md`; two runs identical. 464 MBPP practice problems, greedy decoding, fp32. Summary numbers in `baseline-summary.json`.

| Model | Weights | Whole problems passed | Code tokens predicted (top-1) | Bits per code character | English text tokens predicted |
|---|---:|---:|---:|---:|---:|
| MiniMind | 63.9M | 2 of 464 (0.4%) | 61.0% | 1.08 | 42.6% |
| Pythia-14m | 14.1M | 0 | 40.3% | 1.59 | 25.5% |
| GPT-2 | 124.4M | 0 | 47.5% | 2.17 | 37.4% |

Token accuracy is in each model's own tokenizer and is not comparable across models; bits per character is.

**How whole-problem answers fail.** MiniMind: repetition loop 270, runtime error 117, cut at limit 38, no code 13, syntax error 12, wrong answer 12. Pythia-14m: no code 381, cut at limit 82, loop 1. GPT-2: syntax error 439, loop 12, no code 11, cut 2.

**Selection.** No model reaches 5% at whole problems, so the level is code completion. Pythia-14m (40.3%) and GPT-2 (47.5%) are inside 5% to 60%; MiniMind (61.0%) is just outside. The rule picks the one with fewer weights: **Pythia-14m, judged on code-token prediction.**

**Worth noting, not acted on.** MiniMind is the only model that writes recognisable functions, and 58% of its answers fail by looping, the same kind of failure the reference experiment removed. The frozen rule excludes it here; it is the natural candidate for a follow-up at the whole-problem level.
