# TRANSFER-002 numerical protocol — canonical-affine amendment v1

Status: conditional design; no runtime admission or numerical success claimed. Original native cache/full FAIL remains historical and immutable. `qkv-fixed-two-row-fp32-v1` may progress only after independently reviewed primitive evidence and six-call pilot PASS, exact versioned implementations and runtime admission. CPU FP32/eval/deterministic/2threads unchanged.

## Declared mode and native anchor
Change packed QKV affine evaluation only. For each real row x, form two FP32rows [x,zeros_like(x)], apply original affine weights/bias, keep row0. Fixed two-row work geometry is independent of prefill/cache row count; dummy projection rows never enter attention, positions or cache. Candidate and source-assisted mode-reference integrations are separately authored and cannot share a mathematical helper. No global monkeypatch, FP64forward, unrelated affine/attention/softmax change or prefix recomputation. Native anchor loads original unmodified implementation with no canonical policy/hook. A pilot failure blocks promotion.

## Fixtures and positions
Two separately identified evaluator sets A/B; each real length n in {1,8,32,128}, padding in {none,left,right}. Physical width S=n for none, S=n+4 otherwise. All24 fixtures have four explicit continuation IDs. Masks: n ones plus four zeros on the selected side; append one real mask position per continuation. Position IDs: real prefill tokens0..n-1, pads0; appended real positions n..n+3. Padding uses existing ID0 with mask0; no new vocabulary token. All token IDs0..50303 remain valid inputs; at least one fixture must exercise IDs50277..50303 with numeric undecodable labels. Token bytes and assignments frozen by evaluator before run. Cross-padding invariance is diagnostic, not a replacement for same-fixture gates.


Existing failed token arrays are exposed development material. After candidate/policy freeze, the evaluator must create and hash fresh A/Btoken arrays with unchanged shapes and unmapped-vocabulary coverage, excluding prior exposed complete sequences. No new fixture identity is asserted in this document. The separate verifier owns fixture freeze and original-versus-fresh exclusions.

## Exact full execution schedule
For each24fixture, each mode actor(canonical-reference,independent-canonical-candidate), each arm(baseline,noop,restored): one full prefill S; four cached single-token calls each paired with fresh full prefix S+j,j=1..4. Exactly9calls and5S+14model positions per actor/arm/fixture; sum physical widths1078. Two-mode-actor schedule1296calls,34356positions per role. Each arm has fresh cache and frozen hook semantics.

For each fixture additionally run unchanged native baseline at prefill and four full prefixes:120calls,5630positions per role. No native cached calls or native policy hooks. **Total1416calls/39986actual model-token positions per role; reserve40000positions,1200seconds,9GiB. Both roles2832calls/79972actualpositions/80000reservedpositions/2400seconds.** Dummy affine work is separately recorded as operator work, not secretly sequence tokens; include its measured cost in timeout feasibility. No hidden warmup/inference.

## Unchanged comparisons plus explicit native fidelity
Nine boundaries: embedding, all6block residuals, final normalized state, full50304logits. Require finite arrays, exact real-token argmax and abs(a-b)<=1e-5+1e-4*abs(reference) elementwise. Compare mode actors across all9calls/all3arms:5832boundary rows. Within each mode actor/arm compare each cached token with matching full-pass final real position:5184rows. Compare noop/restored with baseline for all9calls/bothmodeactors:7776bitwise rows. Mode subtotal18792.

For each mode actor/arm, compare each of5full-prefix outputs directly with its unchanged-native baseline:24×2×3×5×9=6480additional boundary rows. Total25272boundary rows per role; finite/top1/cache invariants separately recorded. Native fidelity is full-matrix and all-boundary, never inferred from the small pilot.

Incoming K/V and historical returned prefixes must remain bitwise unchanged with correct lengths/positions; no cache substitution/recomputation. Retain all physical-row arrays, compare real positions under explicit masks, reject nonfinite arrays and retain all50304vocabulary dimensions. No-op/restoration remain bitwise. Cross-padding-invariance diagnostics are distinct, not replacement gates. No interpretation can turn old unchanged-native cache/full failures into PASS.

## Exact raw payload and resource amendment
Mode per role:7,053,699,072boundary bytes +375,349,248returned-cache bytes +164,265,984incoming-cache bytes =7,593,314,304bytes, consistent with runtime/storage-v1.json. Anchor5630rows×205312bytes=1,155,906,560boundary bytes; do not persist native-anchor K/V because anchor is output fidelity only. Total8,749,220,864bytes per role;17,498,441,728both. Preserve original failed producer7,593,314,304bytes; retained raw total25,091,756,032before headers, static/diagnostic/scientific results and acquisition.

Aggregate added-disk cap32GiB, time4800seconds replace24GiB/3600. All other limits unchanged:250000new positions,1239856historical,1489856combined,CPU2,10GiBconservative owned-memory,2GiBfree reserve,7200wallseconds from original admitted start. No refund of failure charges or resetting timestamps. Reconcile live reservations and storage before new runs. Source guard hardcoded limits require independently reviewed update after document review; these numbers alone grant no runtime admission.

## Failure and independent repetition
Any anchor/numerical/restoration/cache/finite failure is FAIL and preserves raw data and full reservation. No row omission, tolerance change or posthoc reinterpretation. At most the bounded diagnosed revision within shared limits; no automatic architecture-wide repair. Separate verifier repeats all1416calls including native anchor with fresh output IDs and compares exact artifacts, arrays, fixture identity, every row and accounting. Reuse unchanged map/static evidence only when identity remains valid; old native arrays cannot substitute for fresh-input anchor. Passing this mode means declared-mode numerical acceptance, not universal equivalence, scientific completion or reversal of original native failure.
