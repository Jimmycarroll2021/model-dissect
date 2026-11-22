# Goal: causal replication audit of public attribution graphs

Set 2026-11-23. Status: COMMITTED, step 1 next.

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

One exploratory data point, `exploratory/ablation.py` and `exploratory/ablation-results.json`: layer-20 transcoder feature 15589, Austin versus Sacramento, 16 Texas/California prompt pairs. Median drop 8.4 logits; random same-size edit about 2; California and non-target prompts 0.0. Not a pass: the feature was inactive on one prompt (H12) and the matched-feature control resolved on 2 of 16. Run in fp32 on GPU, outside the admission process, never independently repeated.

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
