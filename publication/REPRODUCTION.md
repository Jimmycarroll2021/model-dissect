# Evidence, portability and reproduction

This is a curated publication of existing work, not a rerun of model experiments. The original working source was preserved. The publication changes directory configuration, packaging and explanatory links; it keeps numerical tolerances, protocol criteria and scientific outcomes.

## Identity rules

`source-map.json` lists each included upstream file's exact original SHA256/size alongside its publication SHA256/size. `modified: true` means the uploaded bytes differ. **SHA256 values embedded in historical reports/manifests refer to the original experiment, not automatically to the file displayed on GitHub.** Do not feed sanitized manifests into runtime admission. The independent publication reviewer must use the publication commit and manifest.

Machine-specific locations are represented as `EXTERNAL_ARTIFACT_ROOT/...`, repository-relative paths or `PRIVATE_HOME/...` in historical documents. These are labels, not download URLs. The externally retained raw data has not been uploaded and has no promised public download location. Obtain it separately from its custodian, using the existing hash indexes to check identity before recomputing. This repository alone can reproduce the figure from saved metrics and run synthetic checks; it cannot independently re-audit every historical raw array.

The GPT-2 [external evidence index](../evidence/transfer-003/delivery/external-evidence-index-v1.json) preserves original hashes for the corpus, metrics and per-call raw-array metadata. The [full numerical review](../evidence/transfer-003/review/full-review-v2.json), [science review](../evidence/transfer-003/review/science-review-v2.json), and [atlas review](../evidence/transfer-003/review/atlas-actual-v2.json) identify their full evidence chains. Index integrity is not a fresh audit of omitted payloads.

## Obtain model inputs separately

Download only the files required for your chosen experiment, from the exact revision below. Check every file against the linked acquisition record. Do not replace a revision with `main`, substitute a similarly named checkpoint, or treat a model download as runtime admission.

| Model | Pinned upstream | Acquisition/checksum record |
| --- | --- | --- |
| MiniMind | `jingyaogong/minimind-3-pytorch@edba70ec15e06bc4280fbb96ac3383d73a7eab91`, `full_sft_768.pth`; tokenizer files from `jingyaogong/minimind@233d00fa068ec2b8c0d4f07b703934d6391a5a73` | [Acquisition](../evidence/execution/acquisition.json), [allowed manifest](../artifacts/allowed/manifest.json) |
| Pythia | `EleutherAI/pythia-14m@cf967c0a9a04383db6f7b1108d86b2962634b4ac` | [Acquisition manifest](../evidence/transfer-002/acquisition/acquisition-manifest.json) |
| GPT-2 | `openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e` | [Eight-file acquisition audit](../evidence/transfer-003/review/acquisition-v1.json) |

Use official Hugging Face `resolve/<revision>/<filename>` endpoints for model files and raw GitHub paths for pinned MiniMind tokenizer/source files. Review the separate model license. Weights/raw arrays are excluded by `.gitignore`; keep them under an isolated artifact root.

## New runtime procedure

1. Allocate a fresh absolute `MODEL_DISSECT_DATA` directory on suitable storage. Relative overrides, broad system roots, home/checkout/current-directory roots and their ancestors are rejected after symlink resolution. Use a dedicated child directory; artifact child paths cannot be absolute, contain `..`, or resolve outside that root. These checks do not provide an OS sandbox or protection against concurrent filesystem mutation. Reserve CPU/RAM/disk and time according to the applicable protocol; the guards retain limits and exact identity checks.
2. Set `PYTHONPATH=src:.` when using the complete checkout. `pip install -e` installs the core package; reference/runtime scripts and reports remain checkout resources.
3. Verify model input hashes, environment and tokenizer configuration. Historical environment included Python 3.12, torch 2.11.0+cu128, transformers 5.1.0, tokenizers 0.22.2, numpy 2.2.6 and safetensors 0.8.0rc0; these are recorded versions, not an assertion of compatibility with every current release.
4. Use the selected model's spec/contracts to prepare new packet IDs, source/artifact hashes, fixture hashes, reservations and independently reviewed admission. Regenerate all relocation-sensitive hashes. Never replay consumed IDs or old ledgers.
5. Copy required runtime guards/source into the isolated phase layout expected by that protocol, or configure a separately reviewed equivalent. `MODEL_DISSECT_DATA/workspaces/<phase-name>` replaces the old machine root; containment, identity and supervisor checks remain active.
6. Run a bounded pilot before full target inference. Preserve failures and raw arrays. A separate evaluator must check the exact candidate and repeat required experiments. Offline publication tests alone cannot establish target parity.

The original `runtime/guard.py` retains its historical execution-start anchor and is expired for new wall-clock runs. It is retained for inspection; a new MiniMind phase requires an independently reviewed new phase guard, not resetting the old ledger. Synthetic tests patch only the test epoch. The relocated runners intentionally remain gated. There is no one-command new model experiment or automated circuit-tracer integration in this publication.

## Plot-only reproduction

From the checkout, after installing the optional plotting dependency:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=.model-dissect-data/plot-cache python docs/figures/gpt2-plot-final-v1.py
```

This reads the included final metric snapshot, checks the 16 eligible rows and summary medians, and overwrites the PNG/SVG and chart receipt. Use a disposable checkout if preserving the publication commit. Renderer versions can change output hashes; this operation does not reconstruct model logits or prove their provenance.

## Feature-attribution boundary

[Public graph](https://www.neuronpedia.org/gemma-2-2b/graph?slug=thecapitalofthes-1795072198897): one graph, zero interventions. Original labeled export SHA256 `f36a28b2d4802ab6a555fb918f4ee5fcf78a33aaf5a08940706c638e4c5393ad`; the raw export is external. Weight/transcoder revisions are unknown. The [frozen selected claim](../specs/004-feature-attribution-validation/selected-claim-v1.md) and corpus are the scientific authority. Design review receipt original SHA256 `b5e14ab23a50838ef650aaaf63ed1c46914efb25f2e6631358c42f8fa8ca7c67`; frozen manifest original SHA256 `03b42cb68c546709b3da3791f193069f8828e5ad988b78b2c43cd942d8968a73`.

A graph visualization and frozen protocol do not establish a causal mechanism. Local adapter, exact replacement-model identity, fidelity gates and interventions remain blocked/unimplemented as recorded in the protocol.
