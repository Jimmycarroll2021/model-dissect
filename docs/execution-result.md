# Execution result — partial

Independent numerical verdict: **NOT_ACCEPTED_FULL_DEMONSTRATION**. The functional CLI bundle is delivered separately from the numerical core candidate. [Exact runtime verdict](../verification/final-runtime-verdict-v2.json): R0 PASS, R1 PASS, R2 FAIL, R3 NOT_RUN, R4 PARTIAL.

The selected native checkpoint is `jingyaogong/minimind-3-pytorch/full_sft_768.pth` at revision `edba70ec15e06bc4280fbb96ac3383d73a7eab91`, SHA-256 `5b8c49f6c9d965092e651cafeeaeb8705558632b3fd4ac8ab319cbc5a3cbc4a0`. Acquisition downloaded 138166822 bytes across checkpoint, tokenizer and source files; exact origins/hashes are in [acquisition.json](../evidence/execution/acquisition.json). The reference source is pinned at `233d00fa068ec2b8c0d4f07b703934d6391a5a73`.

| Deliverable | Established evidence | Limit |
|---|---|---|
| Inventory and structural map | 91/91 tensors mapped; seven coarse computation claims checked against source in [map-review.json](../runs/evaluation/map-review.json) | Prior-informed graph and supplied settings; not blind discovery |
| Independent forward | PyTorch composition in `src/minimind_re/forward.py`; candidate identity in `analysis/candidate-manifest-v2.json` | Finite tested conditions only |
| Same-mode parity | Initial evaluator reports zero candidate/reference failure rows and zero top-1 mismatches | Does not satisfy all of R2 |
| Cached/full sequence | Initial evaluator: 102 failed rows of 2640, worst tolerance ratio 1.9053236246109009 | R2 FAIL, original `1e-5 + 1e-4*abs(reference)` preserved |
| Internal investigation | R3 NOT_RUN | No probe, causal mechanism or held-out hypothesis result claimed |
| Independent repeat | v2 repeat: 2640 rows, 102 failures; 1296 same-mode candidate/reference rows exactly equal, 192 restoration rows pass | Reproduces failure; R4 PARTIAL |

Initial numerical measurements: [oracle-parity-summary.json](../evidence/execution/oracle-parity-summary.json), raw parity-v1-interface-fix.json (`../runs/evaluation/parity-v1-interface-fix.json`; external or omitted historical artifact). The failures occur in both candidate and reference cached/full comparisons. Independent diagnostic demonstrates shape-dependent FP32 output-projection arithmetic: identical final vectors projected alone versus in a two-row batch differ by 3.814697265625e-6. This diagnostic remains within tolerance and does not prove the sole cause of the larger cached/full failure; differences already occur at the block boundary. Failed reference-only invariance diagnostics are preserved in `evidence/execution/oracle-adapter-check.json`. The adapted reference handles explicit positions, masks and internal observations; parity is against this declared interface, not an unqualified upstream-interface claim.

The reconstruction-v2 change rejects all-masked input rows before computation; numerical expressions were not changed. The independent verifier repeated the full suite on this v2 identity and reproduced the same failure. Candidate manifest SHA-256: `f7c42733df7a08bbe09cacf07c8a3c802a88362bdb627beb26d86a4a5d37d795`. See [implementation-validation-fix.json](../evidence/execution/implementation-validation-fix.json). No threshold was loosened and no model was trained.

The runtime uses `/usr/bin/python`, Python 3.12.3, torch 2.11.0+cu128, CPU FP32, two threads, deterministic mode and TF32 disabled, as recorded in `runtime/environment.json`. The guard uses cumulative token reservations and conservative full-project disk accounting. Earlier smoke evidence lacks CPU metrics and is not retrospectively repaired. The guard is a local process wrapper, not shared-machine isolation or automatic interrupted-run recovery.

All original ceilings remain: 120 cumulative GPU minutes with 40 reserved for independent evaluation; 4 elapsed hours; 1000000 model-input plus generated tokens; 16 GiB peak allocated GPU memory; 8 GiB added project disk; 1 GiB model downloads; zero spend/training. Recorded execution is CPU-only. Current totals and pending work belong to TASK_STATE and runtime receipts, not estimates in this guide.

Model execution has stopped at the diagnosed failure. Remaining outcome: a separately recorded backend/protocol revision with a new freeze and fresh held-out evaluation must resolve R2 before the single controlled R3 hypothesis can run. No additional retry is part of this completed execution assignment. A plan/task checkbox or successful identity check cannot replace those results.

Verifier checkpoint usage: 54732 reserved model tokens, 0 GPU seconds, 148285981 project bytes, 75.58399 ledger CPU seconds; early smoke CPU timing is missing. These are the runtime-verifier checkpoint totals and exclude later non-model CLI/guard checks. Full caps are unchanged.

Functional CLI delivery: `analysis/delivery-manifest-v4.json`, SHA-256 `0256cfcaa150b2946840906c28e1e8a1833e9f52a11846ed7c37a651398c037c`; receipt `evidence/execution/implementation-functional-runners.json` reports 7 passing synthetic tests including actual toy-adapter CLI forward/probe, five arms, exception restoration and refusal of failed R2/identity drift. Target forward core is unchanged. Generic probe raw metrics are UNASSESSED and do not complete R3; target hypothesis selection and norm-matched causal evaluation remain absent.

Independent software review: [software-final-v4-review.json](../verification/software-final-v4-review.json) records PASS_SOFTWARE_SCOPE for the v4 bundle, seven passing synthetic tests and rejection of an alternative configuration before adapter construction for both forward/probe. This fixes the preserved v3 config-binding finding. No target-model forward was performed during software verification; target R2/R3 status is unchanged. Protocol/input identities must remain bound in the surrounding evidence manifest; raw output alone is not complete provenance.
