# Using the delivered GPT-2 dissection evidence

Start with [the result and chart](gpt2-transfer-result.md), then [the exhaustive weight/computation map](gpt2-computation-map.md) and [the cross-model lessons](three-model-transfer-lessons.md). The evidence answers what was mapped, how the editable candidate matched the native model, what one actual input did inside attention, and what the controlled head replacement changed. It does not provide automatic reverse engineering for arbitrary models.

## Inspect without model execution

The final delivery manifest binds the report, source, protocols, corpus, reviews, plots, usage and historical preservation. It deliberately excludes live root task-state hashes: after final audit, a separate closure overlay records only task/state changes and preserves their previous bytes. Review the exact scientific [acceptance receipt](../evidence/transfer-003/review/science-review-v2.json) and [numerical acceptance](../evidence/transfer-003/review/full-review-v2.json); raw paths are recorded there under `EXTERNAL_ARTIFACT_ROOT/workspaces/model-transfer-003-gpt2`.

This command only regenerates the existing final chart from saved metrics. It has no model imports, inference, downloads or installations. Run from the project root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=EXTERNAL_ARTIFACT_ROOT/workspaces/model-transfer-003-gpt2/report-plot-cache PYTHONDONTWRITEBYTECODE=1 python docs/figures/gpt2-plot-final-v1.py
```

The PNG/SVG and provenance receipt are regenerated, so preserve their accepted bytes first if changing plotting tools/versions. Under the recorded environment the script has deterministic data/order/SVG hash salt and output hashes. The final metric snapshot includes all16held-out rows and upstream metric hashes; it is sufficient to reproduce the figure, not a substitute for full raw-control auditing.

## Runtime reuse boundary

Supported computation is CPU FP32 eager, two threads, unpadded/right-padded full-prefill with first key visible. The independent numerical matrix covers real lengths1/8/32/64. Scientific prompts are fixed eight-token unpadded pairs; edits use last-real-token layer8/head8preprojection64-vectors in both actors. Left padding, cache, other backends/dtypes and arbitrary inputs are unverified/excluded.

Use source and contracts as inspectable adapters, not pre-approved runtime commands. Do not rerun consumed packet/admission IDs, initialize over an existing ledger, reuse a prior phase clock or delete failed evidence. A future separately authorized run needs fresh identities, scope/fixtures, readiness, bounded reservations and independent exact admission. No further runtime is required for these accepted results, and no fourth model is scheduled.

## Scientific interpretation

The qualified REFUTED result preserves small positive effects but rejects the preregistered magnitude claim. It does not show the head is useless. Static QK/OV omit nonlinear/contextual computation; derived std uses saved rounded moments. The reconstruction is source-assisted; literal raw-array repetition on the same corpus establishes reproducibility within that scope, not generalization or universal model understanding.
