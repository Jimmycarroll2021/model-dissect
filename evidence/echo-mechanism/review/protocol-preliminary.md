# Independent preliminary echo-mechanism protocol review

Verdict: **PASS_DIRECTION; REVISE_BEFORE_ADMISSION**. The proposed contrast is a meaningful, falsifiable bounded mechanism investigation of observed echo behavior even though task success fails. It is not arbitrary neuron selection: a fixed all-head development scan chooses one component, then untouched cases test that component under bidirectional counterfactual replacement. This is evidence for an intervenable contribution to digit contrast at one receiver position, not a complete source-to-output circuit.

## Independently verified existing evidence

- Reviewed proposal SHA256 `033a6efc07e89b8cc261c5fa6191eb77bd27415ef1ba761db2ef6f2ce2646867` matches the assigned candidate.
- Raw screen SHA256 `87c82eef89d6985f6334cecd3302e0a032210d7af7b0132a1295ca71a42c90cf`: all 16 COPY rows have first four generated IDs `[5078,1226,504,1372]`, recorded text prefix `甲写下数字`, and fifth ID equal to `51 + first input digit`. Joined independently to `fixtures/behavior/screen.json`: all prompt lengths 48, first digit source index 8, expected ID equals generated index 4. This verifies a repeatable recorded local behavior, not a new model run or successful answering.
- Native source `oracle/vendor/model_minimind.py` defaults and `oracle/adapter.py` construction establish 8 layers, 8 query heads, 4 KV heads, head width 96, concatenated width 768. Attention lines 132–133 transpose/reshape then pass the result into a real `o_proj` module. A pre-hook slice isolates one query-head output; a forward hook would instead see mixed output. Actual adapter path is `adapter.model.model.layers[l].self_attn.o_proj`. Runtime tensor shape still requires assertion.
- Budget recomputation: 12000 eligibility + 2496 observations + 53248 scan + 8320 heldout = 76064 positions /1472 forwards per full investigation. Twice =152128, plus4096 parity/control =156224; supplied195792 leaves39568. Continuation protocol independently documents804208 maximum lifetime reservation, hence195792 below1000000. This is reservation arithmetic, not refreshed ledger admission.

## Required fixes

### R1 — Eligibility and denominator handling do not yet define all primary statistics.

Freeze a total scoring function: for any pair failing prefix, digit, margin, finite-logit or positive-denominator qualification, retain raw results and reason, set primary bidirectional recovery score and primary target-minus-control score to zero, count positive-movement success false, and retain fixed denominators 8/16. Undefined raw ratios must be null with reason, never infinity/NaN or silently omitted. Define identical masking for target/control. Actual common-prefix interventions on a naturally divergent case are forced-prefix diagnostics only. Do not substitute its divergent natural history or admit a replacement pair.

### R2 — First 24 shuffled pairs and balance as evenly as feasible are competing, incompletely specified selection rules.

Freeze one deterministic tuple admission/balancing algorithm before inference, exact ordered 8/16 split and complete exclusion manifest covering every prior fixture/response source. Verify uniqueness of all 48 member texts AND token arrays, reject reversed or reused members, freeze first-digit source index and single-token difference. No balancing after outcomes. Publish exact fresh fixtures before any model call.

### R3 — Parity/control allowance is numerically valid but not an executable schedule; pre-search all-head instrumentation checks are unspecified.

Assign the 4096 allowance to exact producer and repeat calls (both native and reconstructed invocations charged), lengths, tolerances and boundaries. Each run must establish reference/candidate same-mode parity, all-layer observer/no-op preservation before search, and hook cleanup including exceptions. Native self-copy/restoration requires exact full-vocabulary logits; cross-implementation parity needs predeclared established tolerances. Record real call shape/device/dtype and assert one cloned 96-vector at [batch0,position51,96*h:96*(h+1)] changed. No bypassing existing runtime guard through direct module calls; native adapter currently creates cache outputs even when input cache is None, so specify actual no-reuse execution semantics explicitly.

### R4 — Control summary and descriptive sign probability are underspecified.

Define control objective as abs(median(min(R_forward,R_reverse))) if that is intended, with R1 masking and deterministic ties. Freeze sign-test statistic, null, direction and treatment of zeros/ineligible pairs before outcomes; report it descriptively, with dependence/common-template limits. A development-selected weakest control is not a random-head estimate and does not show uniqueness. Preserve all head results; held-out cannot trigger reselection.

### R5 — Remaining allowance is a reservation figure, not confirmed completed continuation acceptance or new execution authority.

Before later admission reconcile continuation completion and immutable lifetime ledgers, preserve prior reservations, pin runner/model/tokenizer/source/fixtures and exact per-role budget. Stop if full protocol cannot fit 156224 positions and admitted wall/resource limits. This review grants no runtime admission.

## Interpretation and control adequacy

The one-token prompt contrast, identical other digits, same generated four-token prefix, clean-to-corrupt and reverse patches, same-layer same-width control, self-source no-op and restoration are appropriate for this local hypothesis. Each full forward must start from its declared prompt plus fixed prefix without carried KV cache, prior response text, or leaked hooks. The patch is at receiver position51; it does not identify which attended source positions, keys, queries or values implement the effect. Projected head norms are descriptive only. Large recovery may include nonlinear downstream amplification; ratios can exceed1 and must not be clipped in raw reporting.

Retain thresholds: dev eligibility>=7/8, dev median minimum recovery>=.25 and positive bidirectional movement>=6/8; heldout eligibility>=14/16, bidirectional positive movement>=13/16, median minimum recovery>=.25 and median paired target-minus-control>=.15. Failure is a valid negative/inconclusive result under the specified gate; it must not open another search. A passing result supports this component's counterfactual effect on these fresh echo contexts, not necessity, unique sufficiency, arithmetic, repetition/EOS explanation, or the whole model.

## Execution boundary

No checkpoint, tokenizer, or model was loaded. Zero model forwards / zero model positions. Only assigned review files written. Fixture creation, runtime parity, hooks, head scan, heldout evaluation and repeat remain unexecuted. Preliminary proposal review is complete; admission review requires the revised frozen protocol and exact implementation.
