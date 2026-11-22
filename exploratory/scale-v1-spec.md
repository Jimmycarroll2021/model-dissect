# Scale v1 spec: every featured public graph (frozen before any intervention)

Frozen 2026-11-23. This file, the 22 graph files in `graphs2/` and `audit2.py` are committed before the first intervention run on these graphs.

**Population.** Every graph listed as featured on the Neuronpedia gemma-2-2b graph page on 2026-11-23 with source set gemmascope-transcoder-16k: the 5 pilot graphs (already run, not rerun) plus the 22 in `graphs2/`. The one featured graph built on a different dictionary (clt-hp) is out of scope. No graph is dropped by choice; exclusions happen only through the gates below and are reported.

**Edge selection, model, convention, primary measurement, controls and verdict rules** are exactly those of `pilot-v1-spec.md`, with these additions:

- **Any position.** An edge's feature may sit at any token position. The feature is read and its contribution removed at that position; the effect is measured on the final-position logits.
- **Tokenisation gate.** The prompt must tokenise to the graph's recorded tokens, otherwise the graph is UNTESTABLE.
- **Identity gate** as in the pilot (activation within 1% or 0.05). A graph file that records no activation for a node cannot pass this gate; its edges are UNTESTABLE.
- **Unit gate, per graph.** Take the selected final-position edge in the highest layer. If its hand-computed direct-path effect divided by W is outside 0.5 to 2, the graph's weights are not in the same unit and all its edges are marked UNIT_MISMATCH and left out of the statistics. (Pilot range for this ratio: 0.66 to 1.23.)
- **No variant prompts.** Generality was measured in the pilot only.
- **Specificity** (final-position edges only, reported, not in the verdict): the other-prompt set is the prompts of the other 26 graphs plus the 4 unrelated prompts of the pilot. Reported: fraction on which the feature is active at the final position, and the largest absolute change in that prompt's baseline top-1 minus top-2 gap. SPECIFIC if that change is <= 0.1.

**Summary statistics, fixed now,** over all tested edges from the pilot and this run together: counts by verdict; share with E > 0; median E / W; Spearman and Pearson correlation of W with E; share of graphs in which the edge with the largest W is also the edge with the largest E; verdict counts by layer band (below 20, 20 to 22, 23 to 25).

**Repeat.** Run twice on GPU; result files must be identical.

**Baseline-only pre-check (no interventions), done before freezing:** `dallas-austin` records no activations and has weights in the hundreds, so it is UNTESTABLE; `gemma-it-pirate` does not tokenise to its recorded tokens, so it is UNTESTABLE. The other 20 graphs pass the tokenisation and identity gates on all 60 selected edges.
