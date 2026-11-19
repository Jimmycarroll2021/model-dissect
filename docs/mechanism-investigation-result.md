# MiniMind: an operational internal probe, with an inconclusive copying result

The reconstructed model now has a tested native PyTorch observation/intervention tool: it captures a selected internal contribution, changes one coordinate at one token position, compares controlled outputs, removes the hook and verifies restoration. The first bounded experiment did **not** establish the proposed copying mechanism. Its held-out effect was too small and insufficiently consistent under the preregistered rules. The independent model-execution repeat reproduced all paired held-out rows exactly and the same **INCONCLUSIVE** scientific verdict. The bounded investigation is complete; no further hypothesis search or model runs are admitted. Coordinator accepted the independently verified bounded experiment/tool result; final document review remains separate.

The tested candidate is `analysis/candidate-manifest-mechanism-v1.json` (SHA256 `f851aea7915dd711998f91d15085d2c0f941be715503897676e9f24ef98c8590`), under [frozen MECHANISM-001 protocol](../evidence/mechanism/protocol-v1.md) (SHA256 `b4126dba857be7daed77613665622938260f1d7a67ff031e5f38e6a251404f6a`). This is source-assisted investigation of one identified checkpoint, not blind recovery of its architecture.

## What the tool actually demonstrated

A named identity module exposes the block-0 MLP contribution **after its down projection and before residual addition**. A PyTorch forward hook can capture that tensor or return a modified copy. The experiment changes one residual-space coordinate at the last real input position. It does not identify an MLP hidden neuron or a semantic circuit. Baseline, identity/no-op, target, matched-control and restored runs use identical inputs and full-prefill CPU FP32 execution.

The separately operated reference and new candidate passed all **144 required same-mode comparisons** across12length/padding shapes,3arms and4boundaries, with zero maximum absolute error and zero top1 mismatches. All **225 tool/identity checks** passed, including pristine bitwise agreement and restoration. See [M2 independent result review](../evidence/mechanism/m2-result-review.json) and [raw M2 receipt](../evidence/mechanism/evaluator-m2-v1.json). The separate cross-padding diagnostic still had9failures among32rows; it was reported, not concealed or turned into same-shape failure.

## The single hypothesis and measured outcome

The question was whether one development-selected coordinate selectively supports the correct-versus-foil next-token margin on copying prompts. Baseline observations on16development inputs selected coordinate732 (correlation0.80854), with coordinate27 as the activation-norm-matched control and scale0.876369589005136. The selection was frozen before interventions;32held-out inputs were kept separate. This correlation selected a testable hypothesis, not a discovered mechanism. See [independent selection review](../evidence/mechanism/development-selection-review.json).

Each split used all five arms. Zeroing coordinate732 was compared with subtracting the matched contribution at coordinate27, then removing the hook and rerunning baseline. The16held-out copying items and16unrelated completion items produced:

| Frozen decision measure | Held-out observation | Criterion / interpretation |
|---|---:|---|
| Mean target reduction in expected-minus-foil margin | 0.02898955 logit units | Requires≥0.10; not met |
| Mean target effect minus matched-control effect | 0.03694191 logit units | Requires≥0.05; not met |
| Copying inputs with positive target effect | 11/16 | Requires≥12/16; not met |
| Copying target/control perturbation-norm mismatch | 7.22536% | Must be≤25%; passed |
| Unrelated expected-token NLL increase | −0.02536287 nats | Must be≤0.10; passed |
| Unrelated top1 accuracy drop | 0 | Must be≤1/16; passed, but weak capability evidence |
| Baseline expected-versus-foil preference | 16/16 | Passed narrow task prerequisite; not full-vocabulary accuracy |

The fixed decision rules give **INCONCLUSIVE**, not supported and not refuted. Development was also inconclusive. There is no permission or scientific basis here to choose another coordinate after seeing held-out results. Independent scalar arithmetic reproduced the producer verdict from saved measurements; see [producer-result review](../evidence/mechanism/producer-result-review.json), held-out arm rows (`../runs/mechanism/heldout-evaluate-v1.json.rows.jsonl`; external or omitted historical artifact), and [development review](../evidence/mechanism/development-arms-review.json).

Baseline full-vocabulary copying top1 accuracy was only **1/16**, and unrelated top1 accuracy **0/16**. Thus a stable unrelated accuracy score offers little evidence of preserved useful task capability; the NLL measurement adds information but does not turn these small fixtures into a broad capability evaluation. Expected-versus-foil success alone is not a copying success rate.

## Inspecting and reproducing the demonstrated operation

The local runner is [scripts/mechanism/run.py](../scripts/mechanism/run.py); capture and evaluation logic lives in [mechanism.py](../src/minimind_re/mechanism.py). The executed sequence was `observe` on the frozen development split, freezing the selected-coordinate receipt, then `evaluate` on development and evaluator-owned heldout. Exact packet identities and complete commands are retained in `evidence/mechanism/run-packets/`, `evidence/mechanism/evaluator-heldout-packet-v1.json` and `runtime/ledger.json`.

For example, this is the **historical executed development command**, not an instruction to reuse its immutable run ID or overwrite its output:

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

A fresh run needs fresh admitted run/output identities and live budget/identity checks; copying this command will encounter immutable-run protection. Saved receipts are available for inspection without any model execution. There is no package-install, training or GPU step in this demonstration. The original execution deadline remains1794912024.8448846; it was never reset. Fixture-construction failures and their reservations remain in the lifetime ledger.

## Acceptance boundary

**M2:** independently repeated PASS for this exact same-mode candidate. **M3:** investigation procedure completed, scientific verdict **INCONCLUSIVE**; this is not mechanism-support PASS. **M4:** independent repetition, metric checking and40postrun identity checks passed; coordinator accepted this runtime evidence; final document review remains separate. The [independent execution receipt](../evidence/mechanism/independent/execution-result.json) records220model calls,4640input positions,9.3349seconds wall time and completion before the unchanged deadline. All32paired five-arm rows exactly matched the producer. Lifetime reserved positions are168040, including prior failed attempts, with zero active ledger rows at completion. The old **R2 remains FAIL**, with cached/full execution discrepancy unresolved. Same-mode fidelity is deliberately narrower. This work does not establish universal numerical equivalence, an identified copying circuit, complete understanding of the weights, training history, or performance improvement. Its concrete result is a validated local internal-probing capability and an honest negative-strength result for one frozen hypothesis.
