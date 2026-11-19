# Independent science build review v1

Verdict: PASS_MODEL_FREE_BUILD; target science admission remains conditional on integration gates below. Reviewed the exact science.py 93b231b47c171660a403e5875f818e74eb389e7b1a8893da68772a608560067d and protocol 5e72dbaa6916223862ba31e60a74e65ab5aa15ac043b357230802d528ab0afb5. No target weights/tokenizer loads or target forwards; no implementation changes.

## Established checks

Named author suites independently executed with OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src timeout 120s python -m pytest tests/transfer3/test_science.py tests/transfer3/test_atlas.py -q: 11 passed in 20.54s (4 science plus 7 atlas).

Independent check script and receipt are science-build-v1-check.py and science-build-v1-check.txt. One ineligible development site: 1122 calls, 64 skipped (8 per pair/site), 11 eligible sites. All sites ineligible: INCOMPLETE after 34 calls, no selection and no held access. One held pair ineligible: INCOMPLETE after 1170 calls and persisted excluded-pair metrics, no summary. Heterogeneous 16-pair bootstrap independently rebuilt with scalar estimates and quantiles, exact interval/index-hash match. Original full-schedule test: 1186 calls, 2372 fake token positions, all six arm counts correct, held loader observes selection receipt.

Static review confirms native-only baseline contrast and selection, deterministic tie handling, both directional recoveries and un-clipped P/Q, qualifying sites require all eight pairs, held verdict requires all sixteen, 0.20/0.10/12-of-16 thresholds and candidate P/Q tolerance 0.002. Target and orthogonal controls use each actor's own immutable baseline vectors; FP64 orthogonality and FP32 effective controls validated; edit complement is byte-compared and recorded. Actual matched edited logits and captures receive fixed elementwise/argmax parity. No-op/restoration arrays and logits are byte-compared with baseline. Unpadded inputs make [-1] exactly final-real-token; shape/token/one-difference checks reject unsupported fixtures. Baselines/edits/trace are reloaded from NPZ before metrics. Trace checks actual attention@V against preprojection capture for each layer and both actors, including full boundary retention.

Secondary metrics use stable FP64 full-vocabulary log probabilities and donor-directed KL; bootstrap retains pair-level bidirectional/arm coupling, fixed PCG64 seed, 10000 draws and linear percentiles. Intervals are descriptive for these fixed paired prompts; they do not establish population-level confidence or account for semantic-template dependence. Baseline-derived vocabulary contrasts and development site optimization warrant the protocol's narrow distributional-mediation interpretation.

## Integration/admission gates and limitations

1. Tokenize and hash-freeze exact preregistered text fixtures under pinned tokenizer; algorithm only validates supplied token structure, not provenance.
2. Native/candidate wrapper identity, owned observation source and hook cleanup require separate integration verification. Existing science tests use Fake; inspected candidate tests use two layers. Protocol requires synthetic testing at all twelve science hook sites before target admission. A separate tiny 12-layer/12-head native/candidate test can cover this without target weights.
3. Active guard, hard resource supervisor, live weight/config/source/tokenizer identity callback, and exactly-one-forward trace callback remain caller contracts. Baseline trace layout proof from actual target must pass before interventions; this implementation executes trace first.
4. Independent target repeat, raw-evidence recomputation, unchanged source identity, actual science validity and runtime output caps remain unexecuted. Build PASS is not target admission or empirical head evidence.
5. Skip count is in successful summary; on all-ineligible/held-ineligible failure it is inferable from saved exclusions but not explicitly included in failure.json. No acceptance failure follows: both branches fail closed and preserve exclusions. Failure records could be made easier to audit later.
6. The writer creates exclusive files and hash-checks selection after held execution; it does not provide filesystem immutability against an external writer. Exclusive ownership and frozen caller are required.

Observed storage event: independent script PID4058403 briefly Dl/jbd2_log_wait_commit at ~33s; journal PID1445 also D. Attempted interrupt after coordinator warning found PID already absent; no signal delivered. Script exited zero within 120s and all results above persisted. No new heavy tests launched thereafter. Test completion does not independently clear target storage readiness.
