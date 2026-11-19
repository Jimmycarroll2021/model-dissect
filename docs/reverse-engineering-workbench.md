# MiniMind reverse-engineering workbench

This workbench connects the 91-tensor source-assisted computation map, editable implementation, saved input trace, bounded intervention evidence and the new full-matrix numerical repair. The softmax mode producer passes 9,120/9,120rows; its independent full repeat and raw audit passed. Whole-objective acceptance remains separate. Earlier workbench and all negative results are preserved in versioned delivery snapshots.

## Start with the question you want answered

| Question | Open this artifact | What the evidence establishes |
|---|---|---|
| What is stored in every tensor? | [All 91-key computation guide](model-computation-map.md) and [machine-readable map](../evidence/computation-map/tensor-map.json) | Exact keys/shapes/roles/equations checked against source and metadata; source-assisted, not blind discovery |
| What numerical structure do the weights contain? | [Atlas result](weight-atlas-result.md), readable atlas (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/runs/atlas-v1/atlas.md`; external or omitted historical artifact) and atlas JSON (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-atlas-001-20261118/runs/atlas-v1/atlas.json`; external or omitted historical artifact) | All 91 tensor statistics plus 16 sampled spectra/four representative heads; static geometry is not dynamic feature use |
| How does the computation execute? | [Primitive reconstruction](../src/minimind_re/forward.py), [equations and dimensions](model-computation-map.md), [supplied constants](../artifacts/allowed/supplied_config.json) | Editable tensor operations; separately operated reference validates specified same-mode comparisons |
| What happens for one concrete input? | [Actual-input walkthrough](actual-input-walkthrough.md), [local visual viewer](../visualization/actual-input/index.html) and [independent review](../evidence/actual-input/review/final-review-v1.json) | Eleven fixed input tokens, 64 heads / 704 saved primitive probabilities, eight weighted outputs/residual links and next-token scores; one artificial fixed continuation with observed and derived quantities labelled |
| What is known about cache/full agreement? | [Softmax repair and executable mode](cache-softmax-matrix-result.md), [prior negative matrix](cache-shape-matrix-result.md) and [localization](cache-shape-followup-result.md) | New mode passes 9,120/9,120 unchanged full-matrix rows in both independently repeated runs. Original native and prior mode failures remain historical facts |
| What did it actually generate? | [Longer response result](continuation-result.md) and its linked raw outputs | Four diagnostic responses through 256 tokens, repeated exactly; three incomplete and one wrong under frozen grading |
| Can internal activity be changed reversibly? | [Earlier hook investigation](mechanism-investigation-result.md) and [mechanism runner](../src/minimind_re/mechanism.py) | Capture/no-op/target/matched-control/restoration tools were exercised; the old coordinate hypothesis was INCONCLUSIVE |
| What does the new head experiment show? | [Verified echo result](echo-mechanism-result.md), [independent repeat](../evidence/echo-mechanism/review/recovery-independent-final-v1.json) and [frozen protocol](../evidence/echo-mechanism/protocol-v1.md) | Independent repeat supports bounded head mediation of a local digit preference; no full circuit or source edge identified |
| Which findings remain unresolved? | [Goal/coverage map](reverse-engineering-goal.md) and current state (`../TASK_STATE.md`; external or omitted historical artifact) | Whole-objective gaps, current owners, resources and precise acceptance boundaries |

## Current delivery and numerical limits

The actual-input walkthrough and viewer are separately accepted. They follow saved observations for one artificial fixed continuation, with selected last-token full-prefill activity. The 704 attention probabilities come from saved primitive calculations; derived summaries are labelled. Exact MLP gates/products and the full RMS denominator were not captured. This trace supplies an inspectable input-to-output example, not a new causal or source-edge finding. The separately repeated echo intervention below supplies the bounded causal evidence for a different experiment.

The supported repair candidate is `minimind_re.softmax_bucket_candidate.SoftmaxBucketCandidate`, with the pinned checkpoint and supplied configuration. Its cached path combines current physical query-work geometry with next-power-of-two softmax key width, crops probabilities before unchanged value multiplication, and preserves genuine historical K/V. Native full-prefill arithmetic is unchanged. [The repair report](cache-softmax-matrix-result.md) gives source hashes, backend, exact guarded executable recipe and model-free evidence inspection. It is not a claim for arbitrary models, lengths or backends.

The producer passed all 9,120 rows across 24 fixtures and baseline/noop/restored arms. The independent full repeat and raw audit passed. [The earlier matrix](cache-shape-matrix-result.md) still records24failed rows and its exact repeat; the earlier follow-up and prefix-width observations explain the repair path without erasing failures. Original native cached R2 remains FAIL; the newly declared operational mode is evaluated against the original full comparison requirements.

The authorized narrower M2 result remains144candidate/reference comparisons at zero error. It is separate from the new full-matrix gate. The91-key map, actual-input walkthrough and held-out echo experiment below remain complementary evidence; numerical parity alone supplies no semantic explanation.

## Trace one weight into the computation

Start with `model.layers.6.self_attn.o_proj.weight`. The tensor-map row supplies its shape, input/output dimensions, row-vector equation, atlas row pointer and hashed source locations. The model has eight query heads of 96 dimensions; their concatenated output is 768 dimensional. The output matrix maps that vector back to the 768 dimensional residual space. Head 2 occupies concatenation coordinates 192:288; those coordinates select columns of the stored `[output,input]` matrix when interpreting that head's contribution.

This explains where a head output enters the residual stream. It does not establish what the head attends to, that its contribution is unique or necessary, or that a particular stored scalar has one semantic meaning. Follow the source-backed equations through RMS normalization, rotary position encoding, grouped-query attention, residual additions and the SiLU-gated MLP before interpreting downstream vocabulary logits. [The computation guide](model-computation-map.md) gives all dimensions and normalization/masking conventions.

The following command reads only an existing JSON file and prints that exact tensor record. It performs no checkpoint load, tokenization, inference or artifact writes:

```bash
/usr/bin/python - <<'PYCODE'
import json
from pathlib import Path
root = Path(".")
data = json.loads((root / "evidence/computation-map/tensor-map.json").read_text())
key = "model.layers.6.self_attn.o_proj.weight"
row = next(row for row in data["tensors"] if row["name"] == key)
print(json.dumps(row, indent=2, ensure_ascii=False))
PYCODE
```

Change `key` to another exact name from the 91-row map to inspect its evidence chain. The lookup is metadata inspection; it does not recalculate the atlas or verify checkpoint bytes.

## Move from a mathematical role to an observed effect

The new experiment concerns the first echoed digit after a common four-token response prefix, not the model's final answer to the original task. It compares matched prompts with different digits. The protocol scans 64 attention heads on 8 development pairs, freezes a target and matched control, and tests 16 fresh held-out pairs with bidirectional head-output replacement and no-op/restoration checks.

The development scan selected **layer 6/head 2**, with the weak development-selected **layer 6/head 5** as control. The independently repeated held-out result had 16/16 eligible and positive pairs, median minimum bidirectional normalized digit-logit-gap recovery 0.5534745383939315 and median advantage 0.5535074254291052. The recovery metric is not a percentage of the model explained. **Independent actual repetition passed**, with all 1,480 call-vector hashes and held-out traces matching; the coordinator accepted SUPPORTED_BOUNDED_HEAD_MEDIATION. This supports a counterfactually effective component for this local token preference, not a complete copying circuit, a source-attention edge, necessity, arithmetic ability or an explanation of long repetitive answers.

The raw producer output is here (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-echo-recovery-001-20261118/runs/echo-recovery-producer-v1.json`; external or omitted historical artifact), SHA256 `00d301ddfdefeab2ee927f258624997883a962cc9ab368ff82b18b79ffb786d5`. Read its immutable receipt for the tested runner identity, guard argv, selection and resource accounting. Its 1480 saved vocabulary vectors have 25600 byte storage payloads; this actual run exercised the reviewed serialization fix. The earlier disk-cap failure and its full reservation remain preserved.

This command reads the saved receipt, rather than recomputing the experiment:

```bash
/usr/bin/python - <<'PYCODE'
import json
from pathlib import Path
root = Path(".")
d = json.loads((root / "evidence/echo-mechanism/recovery/producer-execution-v1.json").read_text())
for key in ("status", "target", "control", "heldout_summary", "actual_positions",
            "reserved_positions", "lifetime_reserved_after_producer"):
    print(key + ":", json.dumps(d[key], ensure_ascii=False))
PYCODE
```

## Inspect the reusable tools and execution recipes

The existing [quickstart](../specs/001-minimind-reproduction/quickstart.md) describes the older hook runner and its identity-check command. Its historical generic report/probe status is not the current broad-goal verdict; follow its explicit candidate identity and limitations. `scripts/mechanism/run.py` exposes `observe` and `evaluate` for that frozen earlier protocol. The staged echo runner `src/minimind_re/echo_probe.py` in `EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-echo-recovery-001-20261118` uses its frozen `--packet` interface; the actual argv is recorded in the producer receipt above. This staged runner is experiment-specific, not a general unrestricted head-editing UI.

Model executions are historical recipes or separately admitted work, not inspection commands to paste and rerun. Existing run IDs, outputs, packets and ledgers are immutable. A new execution needs a fresh exact packet/output identity, unchanged or explicitly revised scientific protocol, current resource reservation, runtime guard and independent admission. Reversible edits operate on activation copies/hooks and restoration is measured; do not overwrite original weights to explore an intervention.

Historical accounting is preserved in the respective immutable receipts. The new full-matrix phase reserves101,200positions after1,138,656charged, with final charged total 1,239,856 and zero active runs. This guide is an inspection and execution-interface document, not additional runtime authority.

## What counts as understanding here

The strongest current chain is exact tensor accounting → source-assisted equations → tested same-mode reconstruction → saved input/output observations → controlled internal intervention with separate verification. Each link answers a different question. A sampled spectrum, an attention visualization or an edited output alone cannot replace the other links.

The original native cached R2 remains FAIL. The prior query-shape-only mode failed its complete matrix. The new softmax mode passes the complete producer matrix, with independent repeat and raw audit passed. The old coordinate investigation remains INCONCLUSIVE and atlas-informed hypothesis REFUTED; the newer echo intervention supports bounded head mediation on 16/16 held-out pairs. A complete semantic interpretation of every scalar, full circuit recovery and universal numerical equivalence are not established. Whole-objective acceptance remains for the separate final audit.
