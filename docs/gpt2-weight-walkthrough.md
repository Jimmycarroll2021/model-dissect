# GPT-2: from weights to a measured intervention

Open [the self-contained walkthrough](gpt2-weight-walkthrough.html) in a browser. It works from a local file with no server, installation or network connection. Use the four numbered links, then change matrix blocks, trace token/coordinate and patch coordinate to inspect numeric values. Static explanations, SVG matrix images and expandable numeric tables remain usable without JavaScript.

The page connects four distinct evidence types:

1. **Static atlas:** layer 8/head 8 QK = WQ WKᵀ and OV = WV WO; saved 768 × 768 matrices displayed as 24 × 24 non-overlapping block means (float64 arithmetic mean of each 32 × 32 float32 block). Block averaging can hide cancellation. No coordinate is assigned an invented semantic label.
2. **Actual dev-01 trace:** native clean input “The code is red. The code is”, showing saved Q/K/V, causal attention and captured 64-value head output. The interactive attention-times-V sum is derived from saved arrays, not a new forward pass. Q/K/V axes are batch/head/token/coordinate; head-output axes are batch/token/head/coordinate.
3. **Separate held-01 intervention:** native clean “The door is red. The door is” supplies the donor for corrupt “The door is blue. The door is”. The last token's layer-8/head-8 64 values are replaced; the 6,080 complementary head/position values remain byte-identical. The page shows the exact recipient/donor/control vectors and actual saved final-position contrast logits for both baseline sides and both intervention directions. This is not the dev-01 trace execution.
4. **Frozen held-out finding:** P = mean bidirectional normalized recovery; Q = mean target recovery minus norm-matched orthogonal-control recovery. All 16 pairs are eligible and all 16 Q values positive. Median P **0.08404521678273584 < 0.20** and median Q **0.09451938285766666 < 0.10** preserve the qualified **REFUTED** verdict. The contrast is derived from baseline logits, not a correct-answer score. This does not establish that the head is useless.

Every displayed raw source is identified in [sources.json](../evidence/weight-walkthrough/sources.json) by full path and file SHA256, with array key, dtype, shape and raw-array-byte SHA256 where applicable. Full-resolution arrays remain in `EXTERNAL_ARTIFACT_ROOT/workspaces/model-transfer-003-gpt2`; local raw links require that storage. Token pieces are read from the saved vocabulary JSON; IDs are checked against saved run receipts and corpus entries. No checkpoint weights are loaded.

The design adapts contextualization and small multiples from [Voss et al., Visualizing Weights (Distill, 2021)](https://distill.pub/2020/circuits/visualizing-weights/). Its vision-network feature visualizations are not claimed to transfer directly to GPT-2. Static maps, observed activations and causal interventions remain explicitly distinct.

The underlying science acceptance receipt is [science-review-v2.json](../evidence/transfer-003/review/science-review-v2.json), SHA256 `c39f1bd9a10f318ef3ed91d83fc023b43e3041de69958d5c83e7775f37e8d633`. The existing delivery manifest identity is `f0b097d5183e1f256901b03c0355daaa98602d1e533da5b92bec9816db2af783`. This presentation is a new artifact with a separate [candidate manifest](../evidence/weight-walkthrough/manifest.json); it requires independent review and does not modify the accepted scientific evidence.

To regenerate only this new presentation and its source/check/manifest records, from the project root:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python evidence/weight-walkthrough/build.py
```

This uses existing NumPy and the small saved arrays/metadata, with `allow_pickle=False`. It loads no model, makes no inference calls, installs nothing and writes only the walkthrough's new files. The template and script are retained beside the manifest. [checks.json](../evidence/weight-walkthrough/checks.json) records implementer checks of source identities, token/run alignment, attention × V, donor/recipient baseline links, replacement/complement equality and metric recomputation. Those checks are not the separate independent verdict. The accepted native/candidate parity, restoration and full independent repeat are established by the existing science review; the walkthrough build does not repeat that model execution.

Browser verification is currently limited by the computer-use tool's URL policy: it rejected the local `` page and explicitly prohibited workarounds. No alternate serving or browser path was attempted. JavaScript syntax, embedded dimensions and widget events passed a minimal DOM-stub check; this is **not** a browser rendering, layout or interaction test. The complete denial and check boundary are saved in [ui-checks.json](../evidence/weight-walkthrough/ui-checks.json). The local HTML and this readable Markdown remain available for direct inspection.
