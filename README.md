# model-dissect

Source-assisted transformer dissection: editable PyTorch reconstructions, tensor inventories, computation maps, numerical comparisons, actual-input traces and controlled interventions. This repository collects three scoped applications and a fourth, **design-only feature-attribution protocol**.

**Start here:** [GPT-2 interactive weight walkthrough](docs/gpt2-weight-walkthrough.html) (download/open locally), [readable walkthrough](docs/gpt2-weight-walkthrough.md), [results and chart](docs/gpt2-transfer-result.md), and [lessons across three models](docs/three-model-transfer-lessons.md).

| Application | Delivered evidence | Important boundary |
| --- | --- | --- |
| MiniMind | 91-tensor inventory, editable reconstruction, computation map, input trace, bounded echo investigation and versioned cache repair | Source-assisted; the repaired mode's acceptance does not erase original cache failures. [Workbench](docs/reverse-engineering-workbench.md) |
| Pythia | GPT-NeoX adapter, weight atlas, full-prefill and intervention evidence | Strict cached/full T2 gate remains **FAIL**. [Result](docs/pythia-transfer-result.md) |
| GPT-2 | Full-prefill parity, exhaustive atlas, actual attention trace, matched native/candidate head interventions and independent repeat | Layer 8/head 8 magnitude claim is qualified **REFUTED**; this does not mean the head is useless. CPU FP32 eager, fixed fixtures; no cache/left-padding/general-input claim. [Result](docs/gpt2-transfer-result.md) |
| Feature attribution | One public Gemma graph, frozen claim/corpus/protocol and independent design review | **PREREGISTERED_DESIGN_EXECUTION_BLOCKED**. No causal validation or circuit-tracer adapter implemented. [Selected claim](specs/004-feature-attribution-validation/selected-claim-v1.md) |

## What is included

- `src/minimind_re/`: reconstruction, atlas, diagnostics, guards and scientific metrics; `transfer/` targets Pythia, `transfer3/` targets GPT-2.
- `tests/`: synthetic fixtures and contract/regression checks, without downloaded model checkpoints.
- `oracle/`: separately implemented reference adapters; `oracle/vendor/` contains pinned Apache-2.0 MiniMind source and its license.
- `docs/`, `specs/`, `fixtures/`: readable findings, bounded protocols, scientific thresholds and input fixtures.
- `evidence/`: selected historical acceptance receipts, summaries and identity indexes. These record upstream experiments; they are not proof that this relocated edition reran those experiments.
- `publication/`: provenance mapping and portable reproduction guidance.

Model weights, raw full arrays, environments, caches and operational transcripts are **not in Git**. Follow [evidence and reproduction](publication/REPRODUCTION.md) to distinguish bundled artifacts from externally retained data. Historical hashes are upstream identities; relocated/redacted files have their own hashes in [the publication mapping](publication/source-map.json).

## Use without a model download

Python 3.10+ is required. From the checkout, with dependencies already installed:

```bash
PYTHONPATH=src:. python -m minimind_re --help
PYTHONPATH=src:. python scripts/publication/check_bundle.py
```

Open `docs/gpt2-weight-walkthrough.html` directly in your browser; it is a standalone local document. GitHub displays its source rather than executing its interactive controls.

For a fresh environment, installation is an explicit user operation (not needed for reading):

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,reference,plots]'
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q tests
```

The synthetic tests do not reproduce accepted checkpoint results. They exercise small randomly initialized models, metric arithmetic, malformed inputs and runtime accounting. Reference behavior is version-sensitive: historical versions are listed in [reproduction notes](publication/REPRODUCTION.md).

Set `MODEL_DISSECT_DATA` to an absolute directory on a filesystem with sufficient capacity before a new runtime. The portable default is `.model-dissect-data` under the working directory. The software keeps its original isolation and admission gates; selecting a directory does not authorize or initialize a run. Historical admission IDs and source hashes cannot be reused after relocation.

The generic CLI `report` describes the early MiniMind baseline only. Read the reports above for later results. This is an inspectable research workbench, not an automatic arbitrary-model reverse-engineering service.

## Provenance and licensing

No repository-wide license has been selected for project-authored material. Do not infer a blanket open-source grant from this upload. Third-party MiniMind code remains under its bundled Apache-2.0 license; see [NOTICE](NOTICE.md). Checkpoint licenses are separate, and checkpoints are not redistributed here.
