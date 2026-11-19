# TRANSFER-003 native full-prefill numerical protocol v2

Status: initial frozen design for build review; fixture identity, exact implementation and runtime admission pending. No cache comparison is required by this core protocol.

## Mode and boundaries

CPU FP32 eager, 2 threads, inference/eval, dropout off, deterministic settings and installed library/source identity frozen. Native Transformers oracle and independent editable candidate operate on identical unmodified artifact tensors. Native Conv1D packed QKV/MLP arithmetic and gelu_new semantics must be preserved; no global monkeypatch, source forward wrapper or precision workaround. use_cache=False; past rejected.
15 boundaries exactly: embedding output; block residual outputs 0..11 (12); final normalized state; complete 50257-dimensional logits. Thus 14 hidden-width arrays plus logits per physical token. Retain all physical rows losslessly, evaluate parity on real-token rows under explicit mask, require all stored arrays finite. Compare complete vocabulary; no top-k-only raw data.
Comparator: abs(candidate-reference) <= 1e-5 + 1e-4*abs(reference), elementwise at every real row/boundary; exact real-token argmax agreement. No-op and restored versus corresponding baseline are bitwise at all retained boundaries. Thresholds cannot be changed after results to relabel failure.

## Fixtures and counts

Full core matrix: two evaluator-owned token sets A/B, real lengths {1,8,32,64}, padding {none,right}; 16 fixtures. Padded forms add 4 masked positions. Explicit real positions 0..n-1, pad positions 0; masks are 1 for real and 0 for pad; pad token ID 0, all input IDs in 0..50256. Fixture tokens and hashes freeze after candidate build and before first full matrix; A/B complete sequences distinct. Cross-padding invariance is diagnostic only. No hidden generation/warmup.

Each fixture has two actors (native,candidate) and three fresh arms (baseline,no-op,restored), each one full-prefill call. Exactly 96 calls and 2712 physical model-token positions per role; two roles 192 calls/5424 positions. Boundary actor comparisons: 16*3*15=720 per role; no-op/restoration comparisons: 16*2*2*15=960 per role. Total 1680 boundary comparisons per role plus finite/argmax/digest checks. Zero cache calls.

Raw FP32 boundary payload per physical token is (14*768+50257)*4 = 244036 bytes; 2712 positions => 661825632 bytes per role before headers. Both roles 1323651264 bytes. Deduplication may save identical arrays, but logical manifests must resolve every array byte-exact and independent checks must reconstruct all rows. Resource packet reserves 896 MiB per role for headers and receipts.

Pilot uses four fixtures: length1 unpadded; length8 unpadded; length1 right-padded4; length8 right-padded4. Width sum26; six actor/arm calls per fixture =>24 calls/156 positions and 38069616 raw boundary bytes. Pilot is exposed development material, not substitute for independent full matrix. Stop at120 seconds including serialization/reload; measure peak memory, timing, guard overhead and disk cost.

## Hooks, restoration and intervention equivalence

No-op callback must return unmodified boundary values; restoration run follows real controlled scratch edit restoration and fresh call. Weight hashes and native/candidate source identities must match before/after. Validate head-preprojection hook semantics separately on synthetic tensors before target runtime. At selected layer, preprojection activation shape [B,S,768] is logically [B,S,12,64]; never assume reshape/head order without checking native implementation.
Scientific interventions apply identical position/head edits to native and candidate at this boundary, retaining packed projection arithmetic and full-vocabulary comparisons. Use same comparator for edited output equivalence and bitwise no-op/restoration. Science may not claim candidate intervention faithfulness from baseline parity alone. Bounded scientific case schedule is frozen separately before held-out execution.

## Failure and repeat

Any required parity, finite, restoration, preservation or serialization failure blocks core acceptance and preserves raw evidence. Diagnose using smallest changed test; no new cache-only search. Independent executor actually repeats all96 calls on exact candidate/frozen fixtures under separately reserved packet and recomputes all1680 comparisons, masks, metrics and manifest identity. Optional cached/backend protocol, if later pursued, cannot alter this core verdict or old Pythia failure.

Declared capability schema: full_prefill=true; padding=[none,right]; first_key_visible=true; cached=false; left_padding=false. Left-padding inputs must fail clearly and are not claimed supported. For right-padding science and trace use explicit last-real-token indices, never the final physical pad row; require paired token-position alignment. Natural unpadded prompts are preferred for science. Identical native/candidate masks and explicit position IDs are mandatory.

Review correction v3: both numerical and resource documents reserve896MiB per full-matrix role, above raw661825632bytes. Real pilot serialization must confirm metadata/header margin; fail admission rather than omit arrays. Actual boundary keys are embedding, block0 through block11, final_norm, logits.
