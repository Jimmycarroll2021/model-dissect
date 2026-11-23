# Goal: causal replication audit of public attribution graphs

Set 2026-11-23. Status: steps 1 to 3 done; result in `docs/graph-edge-audit-result.md` (25 graphs, 75 edges). Independent re-measurement, paraphrase check and like-for-like control done. Open: licence and venue.

## The question

Public attribution graphs for Gemma-2-2b (Neuronpedia, circuit-tracer) say "feature X pushes output Y with weight W". Those weights come from a replacement model. This project measures the same edges on the **original model's output logits**:

**How often, and how well, does a graph edge predict the real causal effect in the original model?**

## What we will report, per edge

- Graph edge weight (as published).
- Measured effect: drop in the target-versus-distractor logit contrast when the feature's contribution is ablated at the stated position in the original model.
- Control effect: a random edit of the same size, orthogonal to the feature direction, and a matched-feature ablation.
- Activity rate: fraction of prompts from the same template on which the feature is active at all.
- Specificity: change on prompts where the feature should not matter.

Every edge tested is reported, including edges that fail.

## Why this is worth doing

A prior-work check on 2026-11-23 (three searches, two papers read; not exhaustive) found edge-versus-intervention checks on Qwen with feature-to-feature edges and no size-matched controls (arXiv 2606.28548), and a Pythia study showing ablations flip downstream active sets (arXiv 2611.00606). It found no edge-by-edge test of the public Gemma-2-2b graphs against real model output with controls. If later reading closes that gap, this goal is revised, not quietly kept.

## Starting point

**Correction, 2026-11-23.** The first exploratory runs (`exploratory/ablation.py`, `ablation-results.json`, `results/v1-gpu.json`, `results/v2-gpu-run*.json`, `results/v2-cpu.json`) used the wrong read and write points and are INVALID: they read features from the gained pre-MLP norm output and added the edit to the raw MLP output, before the post-MLP norm. Their numbers (median drops of 8.4 and 6.0 logits, a "Sunnyvale" specificity failure) are artefacts and must not be cited. They are kept only as a record of the error.

The convention was then checked against a published graph (`exploratory/graphs/gemma-fact-dallas-austin.json`): reading features from the pre-MLP RMSNorm output without its learned gain reproduces all 16 layer-20 graph activations to two decimals, and the transcoder reconstructs the post-MLP norm output (relative error about 0.25, against about 800 for the raw MLP output). `exploratory/ablation_v2.py` now uses that convention.

Corrected result for layer-20 feature 15589, Austin versus Sacramento (`results/v2fix-*`, `results/v1fix-gpu.json`): on the 20 held-out pairs frozen in `exploratory/held-out-v2-spec.md`, median drop 0.56 logits (bar 0.5); it beats the random, nearest-feature and size-matched controls by medians of 0.53, 0.54 and 0.67 (bar 0.25) on 80%, 95% and 95% of pairs (bar 75%); California and unrelated prompts change 0.0; the feature is inactive on 1 of 20 Texas prompts. Verdict PASS, narrowly. Two GPU runs identical; CPU agrees to 0.00002. Caveat: the held-out corpus was first scored with the invalid implementation, then rerun after the fix with corpus and thresholds unchanged. On the original 16 pairs: median drop 0.65, same pattern (supporting only).

## Steps

1. Make the one result solid: redesign the matched-feature control (labelled after-the-fact on the original corpus), write a fresh held-out corpus with thresholds frozen before it is run, repeat for identical numbers, compare CPU against GPU.
2. Pilot: 5 graphs, top 3 feature edges each, with a runner that takes an edge spec. The runner must first reproduce the feature-15589 numbers exactly.
3. Go or no-go on scaling to 50 to 100 graphs. A weak or absent relationship is a reportable result.
4. Write-up with code and the full per-edge table, independently rerun before it goes public.

## Rules

- **Exploratory lane:** plain scripts under `exploratory/`. Results are labelled exploratory and never cited as confirmed.
- **Confirm lane:** a one-page frozen spec (claim, corpus, controls, thresholds), one run, one independent rerun.
- Thresholds are fixed before a held-out corpus is scored. No swapping prompts or controls after seeing effects.
- Negative and incomplete results are kept and reported.
- The earlier guard, admission and receipt machinery stays as history; new work does not go through it.

## Not claimed

No complete circuits, no "Texas neuron", no claims beyond the tested prompts, model and dictionary.

## Open decisions (Jim)

- Licence for the repository and whether to publish.
- Venue: forum post, arXiv, or the public repository alone.
