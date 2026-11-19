# Use the versioned MiniMind workbench

The current numerical repair is `cached-softmax-power-of-two-v1`. Its full producer matrix passes 9,120 comparisons; independent full repetition and raw audit passed. Start with the [workbench](../../docs/reverse-engineering-workbench.md) and [repair result](../../docs/cache-softmax-matrix-result.md). The historical mechanism recipe below remains useful for its own frozen experiment and does not select this numerical mode.

## Select the actual implementation

Use `minimind_re.softmax_bucket_candidate.SoftmaxBucketCandidate(weights_path, config_path)` from [softmax_bucket_candidate.py](../../src/minimind_re/softmax_bucket_candidate.py), SHA256 `b9d685f1af23267b7e8c904b01a06e81ae9d7e47fdf71a42688b2b55000c88b3`. Configuration is `artifacts/allowed/supplied_config.json`; checkpoint identity is `5b8c49f6c9d965092e651cafeeaeb8705558632b3fd4ac8ab319cbc5a3cbc4a0`. Within an admitted process the call is `model(input_ids, attention_mask, position_ids, cache=None)`; subsequent single-token calls pass the returned `result['cache']`. Outputs include embedding, block, final, logits and cache. The existing generic CLI default is historical and does not select this class.

The separately authored `minimind_re.softmax_bucket_reference.softmax_bucket_reference(adapter)` context wraps the evaluator's `ReferenceAdapter`; it is a comparison implementation, not called by the candidate. The supported test environment is pinned CPU FP32, torch 2.11.0+cu128, deterministic algorithms, two threads, `MKL_CBWR=AUTO,STRICT`, `MKL_DYNAMIC=FALSE`, batch one and complete masks/logical positions. No arbitrary checkpoint, multi-token cache chunk or cross-backend equivalence is claimed.

## Inspect the exact identity without loading weights

From the project root:

```bash
/usr/bin/python - <<'PYCODE'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('evidence/cache-softmax-matrix/candidate-manifest-v1.json').read_text())
for key in ('candidate_sha256', 'reference_sha256', 'runner_sha256', 'policy_sha256'):
    print(key, manifest[key])
for name in ('softmax_bucket_candidate.py', 'softmax_bucket_reference.py', 'softmax_bucket_matrix_probe.py'):
    path = Path('src/minimind_re') / name
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())
PYCODE
```

## Reproduce through the frozen guard interface

The concrete runner is `python -m minimind_re.softmax_bucket_matrix_probe --packet <packet>`. The exact successful staged recipe, including cwd, guard, policy hash and admission, is recorded in [launch-intent-v1.json](../../evidence/cache-softmax-matrix/launch-intent-v1.json). Its stage is `EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-cache-softmax-matrix-001-20261118`; packet paths are `evidence/packet-producer-v1.json` and `evidence/packet-independent-v1.json`. The [candidate manifest](../../evidence/cache-softmax-matrix/candidate-manifest-v1.json), SHA256 `05be22754d6fdccf48addaeb169fa5a688289d6444172116f2a895b23811f938`, binds their exact bytes.

Those run IDs and outputs are immutable, consumed historical executions. A fresh reproduction uses the same scientific sources/fixtures with a newly frozen stage, packets/output IDs, cumulative resource accounting and independent runtime admission; it must not bypass the guard or overwrite an existing run. The [repair report](../../docs/cache-softmax-matrix-result.md) includes a ready-to-run read-only command for inspecting completed results. Identity inspection and reading saved arrays require no new inference.

## Preserved earlier mechanism recipe

### MECHANISM-001 historical investigation

The historical MECHANISM-001 candidate is `analysis/candidate-manifest-mechanism-v1.json`, SHA256 `f851aea7915dd711998f91d15085d2c0f941be715503897676e9f24ef98c8590`. Native PyTorch hook observation/intervention is implemented and validated on this candidate. M2 same-mode fidelity passed independent repetition. The one selected hypothesis was evaluated on development and heldout inputs; its scientific result is **INCONCLUSIVE**, reproduced with exact paired-row agreement. The bounded experiment is complete; no further model runs or hypothesis search are admitted. Original R2 remains FAIL because cache/full instability is unresolved.

Start with the [investigation result](../../docs/mechanism-investigation-result.md), [independent execution receipt](../../evidence/mechanism/independent/execution-result.json), [structural map](../../analysis/structural-map.md) and current task state (`../../TASK_STATE.md`; external or omitted historical artifact). The result document explains coordinate732,matched control27, the five arms, quantitative decision rules and weak baseline copying capability. Successful tool validation is not discovery of a copying circuit or complete understanding of the weights.

## Inspect the historical mechanism identity without executing the model

Run from the project root with `/usr/bin/python` and `PYTHONPATH=src`. Use a fresh output directory to preserve receipts. This explicit manifest avoids the CLI's historical default:

```bash
inspection_dir=$(mktemp -d /tmp/minimind-mechanism-inspection.XXXXXX)
PYTHONPATH=src /usr/bin/python -m minimind_re check \
  --manifest analysis/candidate-manifest-mechanism-v1.json \
  --output "$inspection_dir/identity.json"
```

Identity PASS checks hashes, not behavior. Behavioral evidence is in the independent execution receipt. The candidate manifest's pre-execution status fields are historical freeze metadata; subsequent signed-by-identity execution receipts establish the measured outcome. The generic `report` CLI still contains the older R2/R3 status summary and is not the current mechanism report.

## The demonstrated guarded operation

The implemented runner is `scripts/mechanism/run.py`, with logic in `src/minimind_re/mechanism.py`. `observe` captures the frozen development baseline and selects one coordinate using the preregistered rule. `evaluate` executes baseline, noop, target, matched-control and restored arms from an exact run packet and frozen hypothesis. This actual mechanism runner uses the new M2 scope; the older generic `probe` interface has different historical R2 admission rules.

The following is the **historical executed development command**, shown to make the demonstrated recipe concrete. It is not a current rerun authorization: its immutable run ID/output already exist and must not be reused or overwritten.

```bash
PYTHONPATH=src /usr/bin/python runtime/guard.py \
  --run-id mechanism-development-evaluate-v1 --role implementation \
  --tokens 730 --timeout 120 -- \
  /usr/bin/python scripts/mechanism/run.py evaluate \
  --protocol evidence/mechanism/run-packets/development-evaluate-v1.json \
  --inputs fixtures/mechanism/development.json \
  --output runs/mechanism/development-evaluate-v1.json \
  --hypothesis runs/mechanism/development-observe-v1.json.hypothesis.json
```

Exact commands/identities are preserved in `evidence/mechanism/run-packets/*.dispatch.json`, the evaluator packet `evidence/mechanism/evaluator-heldout-packet-v1.json`, and `runtime/ledger.json`. Any separately authorized future execution requires fresh run/output identities, exact artifact binding, live accounting and an admitted finite budget. The original deadline1794912024.8448846 was not reset. All authorized mechanism model slots have been consumed.

## Historical execution records — obsolete as current instructions

`analysis/candidate-manifest-v2.json` and `analysis/delivery-manifest-v4.json` describe earlier candidates; they do not identify the then-current hook implementation. Do not use their hash checks as current-candidate checks. Their independently recorded software/numerical receipts remain preserved and valid for those historical artifacts.

The earlier original-gate status was R0/R1 PASS,R2 FAIL,R3 NOT_RUN,R4 PARTIAL, with no selected internal hypothesis and generic probe execution blocked on full R2. That statement describes the closed original protocol. It does not describe the completed, explicitly approved MECHANISM-001 scope, whose selected hypothesis and M2/M3/M4 evidence are linked above. No historical failure is changed to PASS.

Historical evidence includes `verification/final-runtime-verdict-v2.json`, `verification/software-final-v4-review.json` and `docs/execution-result.md`. Source/config exposure remains recorded in `analysis/exposure-ledger.json`; supplied conventions and prior-informed architecture are not blind discoveries. Preserve original checkpoints, tokenizer, failure receipts and frozen protocol. That historical same-mode result did not resolve cached/full or cross-padding sensitivity.
