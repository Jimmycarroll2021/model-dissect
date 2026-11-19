# CACHE-SHAPE-FOLLOWUP-001: bounded layer-one discriminator

Status: PROPOSED_NOT_ADMITTED. Broad user continuation authority is reported by the coordinator; this packet does not create a ledger, clock, reservation or execution permission. Separate exact review and controller admission precede model loading. Scientific goal: distinguish earlier-prefix residual drift from layer-one projection/transform drift; no repair or threshold change.

## Frozen evidence and conditions

The full matrix producer result 9260232e677db11a8c2ede89dc20fc2f1794d41ca8b0ff64c767d5db9b1034b4 is scientific FAIL. Preserve all prior charges and negative artifacts. Original native R2 remains FAIL. Use precisely heldout/length-1-left (failure) and continuation/continuation-length-1-left (passing sibling), baseline only. Freeze exact fixture bytes in fixtures-v1.json. Both physical prefill lengths are 4, real lengths 1, and continuation lengths 4. Do not select different cases after results.

Use unchanged shape-matched-cache-v1 candidate and independently authored shape-cache reference policy, exact current checkpoint/config/vendor identities in candidate-manifest-v1.json. CPU FP32, torch2.11.0+cu128, two threads, deterministic algorithms, MKL_CBWR=AUTO,STRICT, MKL_DYNAMIC=FALSE, GPU disabled. No downloads, training, new weights or tuning. Run actors sequentially and release per-actor state.

## Exact schedule and accounting

For each of two fixtures and each of two actors: one 4-position prefill; four 1-position cached appends; one final 8-position full pass. Exactly 16 positions and 6 calls per fixture/actor; 64 actual model positions, 24 calls per role. Reserve128 per role,256 combined. No primitives or additional model computations in analysis. Independent repeat repeats exactly these calls, comparing every retained array bitwise and derived findings independently. No no-op/restored arm here: observational hook preservation is checked synthetically and each capture's boundary/cache hashes must equal already saved matrix baseline hashes for the matching call.

Assuming completed full-matrix repeat closes at1,137,888 charged positions, proposed ceiling1,138,144 preserves all prior failed charges and adds256. This is a proposal until independently reconciled against terminal ledger. Historical rows are never reset or rewritten; nine prior ledger locks/snapshots must be checked by the eventual additive guard. New phase1800seconds from its first admission; role timeout300seconds, aggregate model wall600seconds. CPU2, owned combined RSS8GiB, new phase512MiB (expected trace data<=64MiB), combined disk8GiB, free reserve2GiB, zero GPU/new download/spend. Guard must measure/reject live resource breaches and retain reservations on any failure. No retry is admitted.

## Capture without numerical mutation

Candidate capture uses Python line tracing on the frozen original callable, copies existing locals only, and removes the trace in finally. It observes layer1 input normalization output; pre-normalization Q/K/V projections; post-normalization/RoPE K; and the full layer0 block output. Native reference uses temporary observation hooks on original layer0 block, layer1 input norm and Q/K/V projection modules. Hooks return None, retain pre-existing hooks and remove only owned handles. The existing independent reference policy remains unchanged. Final returned layer1 cache K supplies the native transformed K observable.

Persist each call's four boundaries and eight-layer returned cache; for this small bounded case storing full-control cache is necessary to resolve the matrix diagnosis's hash-only limit. Persist captured arrays with call metadata, complete masks, physical positions and shapes. Stream each call to disk; keep at most one fixture/actor's path data in memory. Keep raw shapes; comparisons may reshape projected tensors into common physical-token/head coordinates without arithmetic contractions.

## Required analysis and decisions

Reconstruct cached per-token layer0 outputs and normalized/projected operands from the prefill and four appends. Compare final full-pass tensors tokenwise against that reconstruction, including historical physical padding slots. Derive separate exact-byte and original1e-5+1e-4*abs(reference) metrics; top1 remains unchanged.

A. If full layer0 outputs differ at historical tokens, report upstream-prefix drift; latest-token block0 exactness does not refute it.
B. If layer0 outputs match but layer1 normalized input differs, report normalization boundary divergence.
C. If normalized inputs match but projected K/V differ, report projection-boundary divergence consistent with shape-sensitive arithmetic, not a proven kernel cause.
D. If projected K matches and transformed K differs, report normalization/RoPE boundary divergence; the combined boundary does not distinguish those suboperations.
E. If these arrays match, localize using the returned cache and downstream saved boundaries; report unresolved if capture is insufficient. Never invent the cause or adjust tolerances.

All numerical outputs must reproduce the corresponding producer matrix baseline hashes, including the known failed gate. The diagnostic is successful only if captures are complete, unchanged execution is evidenced, all accounting/identity/resource checks pass and independent repetition matches; this does not imply R2 PASS.

## Resource measurement prerequisite

rss-probe-v1.jsonl records a clean process before and after torch import: about12MiB initial and524MiB current RSS after import. Earlier offline diagnosis reported6,993,481,728-byte ru_maxrss but no live samples; its actual peak remains unresolved. Do not waive or subtract that high-water figure as assumed inheritance. The new guard must record its own launch VmRSS/VmHWM and rusage, plus child pre-import equivalents; monitor summed guard+child live RSS every50ms, with missing/malformed measurements fail-closed and no descendants allowed. Record both process VmHWM and rusage at completion and retain the existing conservative final combined high-water rejection. An implausibly high initial high-water is a preflight blocker to investigate, not an exemption. Explicit /proc sampling makes future evidence interpretable without weakening the8GiB guard.

## Preparation boundary

The manifest freezes proposal, fixtures, capture source, synthetic tests and relevant unchanged source/evidence. No follow-up execution guard, model runner, stage, approved packets or admission receipt is claimed by this preparation. Those are necessary implementation/admission work before the proposed64-position diagnostic. Existing guarded matrix runs must not be repurposed.


## Independent design clarification before runtime

Retain layer1 preattention RMS input/output, raw K/V, K-after-headnorm before RoPE, K-after-RoPE and exact positions. Compare historical versus current tokens separately; label all masked physical slots explicitly. Narrow finding concerns K/V provenance only; Q, attention, downstream propagation and logit causality remain untested. Added capture observes candidate K-headnorm return and native k_norm output without modifying tensors. Candidate stores exact RoPE angle and position operands; native cosine/sine operands and candidate actual cosine/sine outputs require an additional read-only observation implementation before runtime. Do not silently call recomputed trigonometric tensors actual observations. This capture requirement is outstanding in the prepared source and must be closed and tested in the next implementation candidate. Distinguish headnorm versus RoPE only with the required pre/post arrays.
