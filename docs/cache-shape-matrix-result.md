# Complete cache-shape matrix result: independently reproduced failure

The shape-matched-cache-v1 mode **fails the unchanged numerical gate on both frozen matrices**. Its earlier successful single length-eight fixture does not generalize to all tested inputs. This is a completed, independently repeated negative experiment, not successful R2 validation. Historical untouched-native cached/full R2 FAIL remains unchanged.

Each role executed both original twelve-fixture sets, all baseline/no-op/restored arms and untouched-native fidelity controls: 24 fixtures, 1,656 model calls, 50,526 actual model positions, 9,120 comparison rows and 9,216 cache-prefix byte checks. Twenty-four rows fail: candidate and independent-reference cached/full logits at continuation step3, in all three arms, for four length-one fixtures:

- heldout/length-1-left
- heldout/length-1-right
- continuation/continuation-length-1-none
- continuation/continuation-length-1-right

Maximum tolerance ratio is1.4585776329040527, against the original elementwise requirement `abs(candidate-reference) <= 1e-5 + 1e-4*abs(reference)`. There are zero top-1 mismatches and zero nonfinite rows. Those observations do not override the elementwise failure. The other9,096 rows pass their respective comparisons; they do not establish universal equivalence.

The [independent final review](../evidence/cache-shape-matrix/review/independent-repeat-final-v1.json) records `PASS_INDEPENDENT_EXACT_REPEAT_OF_SCIENTIFIC_FAIL`. It independently checks9,120 rows and matches18,144 boundary/chunk tensors bitwise, with identical call journals. Producer and independent result JSON files have the same SHA256: `9260232e677db11a8c2ede89dc20fc2f1794d41ca8b0ff64c767d5db9b1034b4`. Review SHA256 is `24c565a10005660b9d952045990fae9a25f32e4d2b1bdba64f7873022acae738`.

Producer and independent wall times were87.8083 and89.1966seconds. Each retains its50,600-position reservation; historical charges plus both roles total1,137,888, with zero remaining under that ceiling. Both guard rows are completed. The completed phase footprint was4,035,503,152bytes, below4GiB, according to the independent verifier's terminal audit; resource detail and exact ledger identity are retained in the final review and audit. No retry, repair, training, download or GPU execution is included.

All four boundaries per call and lossless cached-path K/V chunks are retained under `EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-cache-shape-matrix-001-20261118/runs/`, in producer and independent directories. Full-control unused K/V arrays were not retained under the frozen protocol. This bounds later causal diagnosis: hash divergence cannot by itself identify the differing token or operation.

The [offline diagnosis](../evidence/cache-shape-matrix/diagnosis/findings-v1.md) is subsequent implementer analysis, not part of the independent matrix acceptance. It finds layer-one cache hash divergence despite exact current-token block-zero suffixes, also in passing siblings. A [small proposed discriminator](../evidence/cache-shape-followup/protocol-v1.md) would capture earlier-prefix states and layer-one projection boundaries. That proposal is not admitted or executed and must not be reported as a resolved cause.
