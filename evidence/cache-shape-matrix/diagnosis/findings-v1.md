# Offline diagnosis of complete matrix failures

Status: saved-array localization, not causal proof or a repair. Producer result SHA256 9260232e677db11a8c2ede89dc20fc2f1794d41ca8b0ff64c767d5db9b1034b4. Only producer evidence examined; independent repeat is separate.

## Observed

Four failing length-one fixtures and two passing length-one siblings were examined, baseline only, candidate and independently implemented reference. For all six, cached embedding and block-zero output suffixes are bitwise equal to the corresponding full-pass suffix at all four continuation steps. Candidate and reference saved cached boundaries match each other. The earliest saved final-representation divergence is step 1 for both unpadded fixtures and step 3 for all four padded fixtures. Both passing siblings also show this divergence; drift alone is not sufficient for gate failure.

Reconstruction of initial-plus-append cached K/V chunks reproduces the saved cached full-cache hashes. Comparing those reconstructed hashes with the full-control journal hashes shows layer 0 K/V exact throughout. Earliest differing whole-cache K and V occur in layer 1, at the same respective steps. Full-control cache arrays were deliberately omitted, so these hashes cannot locate the differing token or element. A different historical token could explain a whole-cache hash difference even when the current block-zero suffix is equal.

All four failing fixtures first cross the original numerical acceptance threshold in final-step logits, one vocabulary element per actor/arm; the first differing saved representation is earlier than the failing output gate. Final representations remain within the original elementwise tolerance. Producer reports zero top-1 disagreements and zero nonfinite rows; those do not override its numerical FAIL.

## Interpretation and evidence boundary

Shape matching attention query contractions repaired the previously exposed length-eight fixture, but does not guarantee whole-matrix equality. The present evidence localizes an unresolved divergence to layer-1 cache production or earlier-prefix state feeding it, before downstream final logits. It does not prove a layer-1 attention contraction defect. Candidate source still uses one-row Q/K/V projections and normalization in cached mode, whereas full passes use all rows. Different projection kernels, normalization/reduction behavior, or earlier full-prefix block states are competing explanations. No weights, tokenizer, model forward or matrix contraction was loaded/executed by this offline task.

## Smallest next discriminating test (proposal only; NOT executed)

Use heldout/length-1-left (fails) and continuation/continuation-length-1-left (passes), baseline only. Each has physical prefill length 4. For each actor (candidate and reference), run prefill 4 + four single-token cached appends + one final full pass length 8: 16 positions and 6 calls per actor/fixture. Both fixtures and actors total 64 actual positions / 24 calls per role. Provisionally reserve 128 positions per role, 256 combined if independently repeated; derive exact frozen schedule before any admission. No sweep, tuning, intervention or acceptance replacement is proposed.

Capture full layer-0 output for every physical token and layer-1 normalized input, unnormalized K/V projections, normalized/rotated K, and final cache K/V. Compare final full pass against reconstructed cached path token by token. If full layer-0 outputs differ for historical slots, localization moves upstream; if all corresponding inputs match but projected K/V differ, projection arithmetic is implicated; if projections match and transformed K differs, normalization/rotation is implicated. Retain existing boundaries and exact finite/tolerance controls. No hypothesis may be promoted from hash mismatch alone.

Estimate: sequential tiny fixtures, 2 CPU threads, <=300 seconds per role, proposed <=64 MiB new trace storage. Model-runtime RSS requires its existing separately reviewed admission envelope; do not substitute this offline task's 2 GiB limit. A fresh isolated diagnosis must be reviewed for exact accounting and independent verification, not silently taken from full-matrix acceptance reservations.

## Resource limitation of this offline task

Completed in approximately one second. Python ru_maxrss reported 6,993,481,728 bytes, identical to the producer guard high-water. Current RSS was not sampled during tensor processing, so this result does not establish compliance with the assigned <=2 GiB incremental memory bound. A subsequent plain-Python baseline reported about 12 MiB, but cannot retrospectively resolve the discrepancy. No further tensor work was run after observing this. Preserve the high-water value as reported; do not present a confirmed lower peak. Numerical data remain available for independent review.
