# Using the two-model investigation workbench

This workbench helps inspect what weights store, follow an editable computation, compare it with a separate reference, and test a reversible internal intervention. Its method transfers through explicit architecture adapters and measured acceptance. It does not automatically dissect an arbitrary checkpoint or establish the meaning of every scalar.

**Documentation snapshot: transfer-002 report draft v1.** MiniMind has a separate accepted whole-objective result. Pythia has a measured76-tensor atlas and5832zero-error cross-actor comparisons, but strict cache/full T2 FAILED. The full-prefill producer completed432/432baseline checks and reports SUPPORTED_BOUNDED_BLOCK_EFFECT at block4 on16/16held-out pairs. Independent repetition is IN PROGRESS and final acceptance remains pending. Read the current result draft (`pythia-transfer-result.md`; external or omitted historical artifact). Final scientific results and independent integrated verdict must be linked before delivery is called complete. Live work may have advanced after this snapshot; use exact result receipts rather than infer success from available packets.

| Question | MiniMind evidence | Pythia transfer entry point |
|---|---|---|
| What does each tensor do? | 91-key map (`model-computation-map.md`; external or omitted historical artifact) | 76-key GPT-NeoX computation map (`transfer-002-computation-map.md`; external or omitted historical artifact) |
| Where are the editable equations? | Versioned accepted implementation (`../src/minimind_re/softmax_bucket_candidate.py`; external or omitted historical artifact) | GPTNeoXCandidate (`../src/minimind_re/transfer/gpt_neox_candidate.py`; external or omitted historical artifact) |
| How do I inspect static structure? | Atlas result (`weight-atlas-result.md`; external or omitted historical artifact) | Atlas implementation (`../src/minimind_re/transfer/atlas.py`; external or omitted historical artifact), admitted static packet/results |
| What numerical claims are established? | Whole-objective independent review (`../evidence/goal-coverage/whole-objective-review-v2.md`; external or omitted historical artifact) | Numerical protocol (`../specs/002-model-transfer/numerical-protocol.md`; external or omitted historical artifact); require producer and independent execution receipts |
| Can I follow one input and edit internal activity? | Actual input (`actual-input-walkthrough.md`; external or omitted historical artifact), bounded echo result (`echo-mechanism-result.md`; external or omitted historical artifact) | Block intervention proposal (`../specs/002-model-transfer/internal-protocol-proposal.md`; external or omitted historical artifact); trace/scientific execution still pending in this draft |

MiniMind's accepted result is source-assisted and specific to its checkpoint, CPU environment and declared cached-softmax numerical mode. Its native cache/full failures, rejected repairs, earlier inconclusive coordinate experiment and refuted atlas hypothesis remain preserved. Pythia's success cannot be inferred from that record.

## What changes between the two architectures

MiniMind's mapped computation uses RMS normalization, its rotary conventions, grouped-query attention and a SiLU-gated MLP. Selected Pythia-14m uses six GPT-NeoX blocks, hidden128, four32-dimensional heads, LayerNorm with offsets, per-head packed Q/K/V projection, partial RoPE on8head dimensions, parallel attention/MLP residual addition, GELU and separate input/output embeddings. Its full50304vocabulary rows remain present, including27rows without tokenizer entries. Undecodable rows are numeric IDs, not omitted output dimensions.

The Pythia candidate implements tensor primitives and imports no Transformers forward helpers. Its separate native reference (`../src/minimind_re/transfer/reference.py`; external or omitted historical artifact) loads installed Transformers GPTNeoX eagerly with local files only and trust_remote_code=False. Reference agreement is an independent numerical check, not proof of unique recovery or a universal model explanation. Source assistance is explicitly disclosed.

The candidate's NeoXConfig rejects non-GPT-NeoX model_type, non-GELU activation, nonparallel residuals, tied embeddings, scaled RoPE/sliding windows and invalid dimensions. Weight keys/shapes must match its computed architecture map exactly; unexpected or missing keys fail. These checks define a supported implementation subset; a config passing them still needs numerical validation for its particular checkpoint. No universal unsupported-architecture handling is claimed beyond explicit rejection at these boundaries.

## Inspect saved metadata without running a model

From the project root, this command uses only the standard library and reads existing metadata. It does not import PyTorch, tokenize or load checkpoint tensors:

```bash
/usr/bin/python - <<'PYCODE'
import json
from pathlib import Path
root = Path('.')
for name in ('evidence/transfer-002/acquisition/acquisition-manifest.json',
             'evidence/transfer-002/runtime/numerical-producer-v3.packet.json',
             'evidence/transfer-002/review/admission-numerical-producer-v3.json'):
    path = root / name
    print(name)
    print(json.dumps(json.loads(path.read_text()), indent=2))
PYCODE
```

An acquisition manifest identifies files; an admission permits one exact bounded execution; neither establishes successful outputs. Inspect the run's supervisor receipt, append-only ledger and output summary together. The v3 packet's phase root is `EXTERNAL_ARTIFACT_ROOT/workspaces/model-transfer-002-pythia-14m/phase`; its output path is recorded in the packet. Missing or interrupted output remains unverified.

## The actual guarded CLI

The implemented entry point (`../src/minimind_re/transfer/run.py`; external or omitted historical artifact) accepts `--packet PATH --admission PATH`. Its current stages are `static`, `numerical` and a model-free `synthetic` integration exercise. It does not currently expose the design's proposed `capture`, `intervene` or `report` CLI verbs. Do not treat proposed names as runnable commands.

The following shows the concrete v3 numerical invocation shape from the current packet/admission files. It is an execution recipe to audit, not a command to rerun a consumed packet:

```bash
cd '.'
PYTHONPATH=src /usr/bin/python -m minimind_re.transfer.run   --packet evidence/transfer-002/runtime/numerical-producer-v3.packet.json   --admission evidence/transfer-002/review/admission-numerical-producer-v3.json
```

The static counterpart uses `static-producer-v3.packet.json` and `admission-static-producer-v3.json`; the independent numerical counterpart uses `numerical-independent-v3.packet.json` and its matching admission. The launcher validates environment versions and admission packet hash, supervises the owned process and rejects an existing supervisor receipt. The child stage reserves through PhaseGuard and enforces identities, calls and limits. Do not invoke its hidden `--child` option directly.

For a **new reproduction**, create fresh run/output IDs and reviewed packets, reconcile all prior and active ledger charges, reserve resources and obtain independent admission bound to exact packet/code/artifact bytes. Never copy a PASS verdict onto a changed packet, overwrite completed output, delete receipts to force a rerun, or treat failed reservations as refunded. Historical MiniMind charges remain1239856; transfer's250000-position allocation has combined ceiling1489856. The current numerical packet reserves34400positions and900seconds per role. Other stages share the same aggregate transfer allocation; do not budget each independently at the full ceiling.

## Capture and reversible edits in the existing API

Within an already admitted process, `GPTNeoXCandidate.from_directory(directory, expected_hashes=...)` verifies config/checkpoint identity and loads tensors. This constructor's hashes do not grant runtime admission. `forward(input_ids, attention_mask, position_ids, past_key_values=None, use_cache=True, boundary_hook=None)` returns embeddings, block outputs, final normalized state, logits and cache. Use full historical masks and correct logical positions for padded cache continuation; never reuse a cache from a different model or intervention arm.

`boundary_hook(layer_index, hidden)` operates at a block output; the lower-level `boundary_transform(name, tensor)` exposes additional named boundaries. Callback outputs preserve shape, dtype and device. The native reference supplies a corresponding block-output hook and removes handles in a finally block. These are native PyTorch mechanisms, not an integrated third-party interpretability platform. A hook's existence is not evidence it was scientifically validated: numerical no-op/restoration, controlled target/control arms and independent repetition establish that evidence.

The scientific proposal changes a single last-position128-dimensional block vector, compares donor replacement with an equal-norm orthogonal control, and freezes selection before heldout. It explicitly does not intervene on an individual head. Scientific execution uses the separate implemented `minimind_re.transfer.science_full_run` entry point; the numerical CLI does not silently perform it. It requires `--packet PATH --admission PATH`, exact scope `T2S_FULL_PREFILL_STRICT_T2_REMAINS_FAIL`,12500reservedpositions,300seconds and1GiB per role. Concrete packet paths are `evidence/transfer-002/science/science-full-producer-v1.packet.json` and corresponding `evidence/transfer-002/review/admission-science-full-producer-v1.json`. These identify an active/consumed execution, not a rerun instruction. The current supported scientific scope uses original full-prefill calls only and never reuses cache. Fresh reproduction needs new IDs/reservations/exact admission; do not invoke hidden --child. Candidate intervention-equivalence is not tested: scientific edits run on native block hooks, while candidate baseline fidelity is checked separately.

## Which researcher tools are actually integrated

The tool landscape (`tool-landscape.md`; external or omitted historical artifact) is a survey, not an installation or integration receipt. The implemented paths here use PyTorch tensor operations/hooks, Transformers as separate native reference, safetensors loading, tokenizer artifacts and local evidence/guard code. Do not infer that every catalogued package, REA capability, activation dashboard or third-party circuit tool has been installed, connected or validated against these checkpoints. Static weight statistics, dynamic captures and controlled causal evidence answer different questions.

## Adding a third model

1. Select a licensed, bounded artifact; pin revision, config, tokenizer, weights and safe loader. Record all source assistance and artifact exposure.
2. Compare architecture semantics with existing adapters. Reuse an adapter only when supported config, exact tensor schema, masking, positions, normalization, attention, MLP and cache semantics match. Otherwise implement a small new independent adapter; retain existing guards and frozen artifacts.
3. Produce exhaustive tensor/alias accounting and a computation map. Preserve all vocabulary rows, even those without decoded tokens.
4. Operate an independent native reference and freeze numerical fixtures, internal boundaries, cache/full checks, no-op/restoration, tolerances, backend and resource/storage counts. Passing constructor checks or a synthetic test is insufficient.
5. Execute producer and independent numerical verification, then preregister one bounded intervention with disjoint development/heldout, matched controls, honest negative/inconclusive rules and actual independent repeat. Trace one input from tokens to saved internal/output values.
6. Freeze an integrated manifest and get separate whole-objective verification. Report exactly what transferred, what needed architecture work and which gates remain incomplete.

Two implemented architectures establish a useful transfer test only when both measured acceptance chains are complete; they do not establish arbitrary-model support or complete semantic understanding.
