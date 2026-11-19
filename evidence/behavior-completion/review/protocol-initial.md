# Independent protocol triage — BEHAVIOR-002

Status: HOLD pending frozen protocol and exact implementation. Read-only prior-evidence inspection; no checkpoint loaded or model executed.

Prior evidence: `evidence/behavior/independent-screen-review.json` establishes exactly 0/16 full-vocabulary first-token correct per family and exact 48-row independent repeat, explicitly not complete-answer performance. A complete-response measurement is a legitimate distinct measurement; reuse of known screen rows is disclosed selection evidence. Qualification rows must remain unobserved until selection.

Required blocking checks before runtime:

1. Freeze a conservative answer grammar before output inspection. Numeric strings embedded in arithmetic, repeated numbers, negation, multiple candidates, contradictory comparisons, signed/decimal/multidigit values and echoed question operands cannot be counted by taking any convenient final digit. Report parser rejection separately from model error. A conservative grammar can undercount actual semantic correctness; disclose that limitation.
2. Distinguish semantic answer accuracy and strict requested-only-token compliance. Threshold must name which metric selects/qualifies. Both require actual termination (recognized pinned EOS/EOT) rather than cutoff or merely a parsable prefix. Special token handling must preserve stopping evidence; do not let decode(skip_special_tokens=True) conceal an internal control token.
3. Reserve and count every full-prefix forward: for length L and n generated tokens, positions=n*L+n*(n-1)/2 (including forward producing EOS). No extra uncharged stop/detection forward. Maximum 24 generated includes EOS. Bound contextual input positions after concatenation and use cache=None for every full recomputation.
4. Match old frozen prompt IDs and tokenizer API bytes exactly, with pinned tokenizer special IDs/template/options. Greedy argmax uses full vocabulary, finite logits, deterministic lowest-ID ties, no EOS suppression, answer prefill or forced labels.
5. Select first family passing >=13/16 once, then exactly its 32 untouched qualification rows >=26/32 once. No fallback after qualification failure. Independent repeat covers all 48 screening rows and conditional 32 qualification rows, including failed/truncated outputs, exact tokens, stop reasons, parses, counters and thresholds.
6. Guard must reconcile BOTH historical reservations (original + atlas = stated 295600) with reachable source identities/locks; copying one historical snapshot does not serialize against another active runtime. Reserve repeat positions/time before producer admission. No failed-slot refund or clock reset.

Expected independent synthetic tests: answer grammar adversaries, EOS at first/last permitted token, cutoff with a plausible answer, emitted non-EOS special token, zero/NaN/Inf logits rejection, tie handling, cache-none full-prefix sequence lengths and exact position formula, 12/13 screen boundary and 25/26 qualification boundary, first-eligible deterministic selection, failed qualification no fallback, tampered identities and missing repeated outputs.

No actual runtime admission is established by this review.
