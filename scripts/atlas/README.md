# Static weight atlas CLI

This module is model-free: it loads a CPU tensor dictionary with `weights_only=True` and performs bounded static tensor operations. Actual checkpoint execution requires the ATLAS-001 reviewed runtime launcher; the command below is an interface example, not runtime admission. Never invoke it directly on the original checkpoint outside the guard.

```bash
PYTHONPATH=src /usr/bin/python -m minimind_re.atlas \
  --checkpoint /admitted/path/checkpoint.pth --expected-sha256 CHECKPOINT_HASH \
  --config /admitted/path/config.json --config-sha256 CONFIG_HASH \
  --tokenizer /admitted/path/tokenizer.json --tokenizer-sha256 TOKENIZER_HASH \
  --output /admitted/path/new-report \
  --gram-dim 128 --spectral-matrices 16 --max-seconds 300 \
  --directions 0 1 --head 0:0 --head 7:7 --token-ids 0 1 2 3
```

Freeze actual paths, hashes, directions, heads, token IDs and limits in the reviewed run packet first. The sampled matrix selection is deterministic: evenly spaced indices among sorted 2D tensor keys, up to16; each matrix uses evenly spaced row/column indices, at most128 each by default. All indices and scopes are saved. Top32 singular values are emitted. Rank is threshold-dependent **submatrix numerical rank**, not original full-matrix rank. `--full-gram` is a separate capability, not admitted by the default protocol.

JSON contains every tensor, scalar statistics including nonfinite/zero accounting and population standard deviation, aliases, selected spectra, scalar comparisons across layers, selected token-row cosine geometry, raw selected-head QK and factored OV projections. Markdown displays the complete tensor table, compositions, geometry, comparisons and dataflow. Manifest binds reports and checkpoint/implementation provenance. Optional config/tokenizer inputs require expected hashes, checked before/after use. Tokenizer labels are raw JSON pieces, not decoded text. No feature semantics or causality is inferred.

The existing source-assisted architecture uses contiguous GQA query groups. Raw QK excludes RMS normalization and RoPE; it is **not attention**. OV excludes attention mixing, final normalization and other layers. Gated SiLU MLP cannot be collapsed into a single linear map.

Elapsed checks are cooperative between work units, and measured atlas time excludes input/output hashing/load. The external reviewed launcher must enforce the full process wall/memory/disk budget. Spectral work estimates are conservative scalar-work proxies, not measured FLOPs. Full scalar statistics cover all tensors; nonfinite tensors fail affected spectrum/composition analysis with explicit status. Unsupported layouts/types are retained with reasons. Output directories must not already exist.

Synthetic validation (does not load the real checkpoint):

```bash
PYTHONPATH=src /usr/bin/python -m unittest discover -s tests -p 'test_atlas.py' -v
```

Saved implementer result: `synthetic-tests.txt`. Separate exact-candidate verification and actual91-tensor execution remain pending.
