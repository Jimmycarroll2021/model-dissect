# Full-matrix softmax cache repair

Status: independently verified PASS for the new mode on both complete matrices. Whole-objective acceptance is separate.

The new `cached-softmax-power-of-two-v1` mode passes all 9,120 producer comparisons on both original twelve-fixture matrices. It uses the exact original elementwise tolerance and logits top1 requirement. This repairs the observed numerical acceptance failure for the producer's full tested scope; the independent repeat and raw audit confirm the result. The historical native-cache failure and prior query-shape-only matrix's24failed rows remain immutable evidence.

## What changed

The candidate already expands cached query work to the current full physical sequence width, retaining dummy historical queries while computing only new Q/K/V. The additional rule expands only the cached softmax key dimension from current physical countT to the next power of twoB: fill new columns with -Inf, preserve existing causal and finite padding scores, perform FP32 softmax, then crop firstT columns contiguously before the unchanged weighted-V contraction. At powers of two, it uses the existing softmax call. B depends only on currentT, never fixture ID or future length. Full-prefill arithmetic and the untouched native adapter remain unchanged. Historical K/V is reused and checked for byte preservation, not recomputed from the prefix.

Saved natural operations localized the earlier difference to softmax at equal common masked scores, with masked future columns. A separately repeated model-free replay supported this one fixed policy. This is finite-precision arithmetic alignment on a fixed backend, not proof of a mathematical attention defect or universal cache/full equality. Width buckets change at their boundaries; successful tested cases do not establish arbitrary-length guarantees.

## Tested scope and evidence

Each full role executes24fixtures ×3arms (baseline, real clone-hook noop, restored),1,656model calls and50,526actual positions. It evaluates9,120rows:5,280original comparisons,2,880untouched-native full-fidelity comparisons,960native restoration comparisons. Real lengths1/8/32/128, none/left/right padding, four cached appends and every full-prefix comparison remain. Comparator: atol1e-5 +rtol1e-4*abs(reference), all real positions, finite gates and exact logits top1. No threshold, fixture or arm was removed.

Producer result:9,120PASS/0FAIL;9,216KV checks each check unchanged input and returned prefix;72full-boundary archives and720lossless initial/appendKV archives retained. Runtime87.46304110100027seconds. Result SHA256`b69300eb1d3887d45474112fa96d492ef83c13a853033e0f3c5522c18ce5cfec`.

- [Frozen candidate manifest](../evidence/cache-softmax-matrix/candidate-manifest-v1.json), SHA256`05be22754d6fdccf48addaeb169fa5a688289d6444172116f2a895b23811f938`.
- [Exact admission](../evidence/cache-softmax-matrix/review/runtime-admission-v1.json).
- [Producer terminal receipt](../evidence/cache-softmax-matrix/producer-terminal-v1.json).
- [Independent small-screen result](../evidence/cache-softmax-bucket/review/independent-repeat-final-v1.json):80rows and560saved tensors repeated exactly.
- [Prior negative matrix](cache-shape-matrix-result.md) and [earlier localization](cache-shape-followup-result.md) preserve the rejected mode and diagnosis.

## Independent confirmation and tolerance margin

The [final independent review](../evidence/cache-softmax-matrix/review/independent-repeat-final-v1.json), SHA256 `956615edb52d3017d9f43b36bbc497733960e5d3633f9b936348d4f3a4c03e8a`, confirms both complete runs. All **18,144 saved boundary/chunk tensors** match byte-for-byte between producer and independent repeat; all 1,656 requests and 9,120 comparison-row metrics agree. Per role the raw audit checks 6,624 boundary hashes, 11,520 cached tensors and 9,216 returned-prefix reconstructions, with no failures. Runtime separately asserted input-cache preservation.

**Tolerance agreement is distinct from repeatability.** The worst numerical comparison reaches **0.9981775283813477 of the permitted tolerance**, close to the threshold. Untouched-native full-fidelity comparisons have maximum ratio **0.0**. All rows pass, with zero top-1 mismatches and nonfinite failures; the cached/full outputs are not claimed byte-identical. Exact bytes above concern repetition of the same execution mode across the two runs. Preserve the pinned backend and do not generalize this margin to other environments.

Independent result SHA256 `f66d40fb47884dca50b3e0035ad2e4bd0399e72ec88e2bbf407e427d1605a79b`; independent duration 89.191945 seconds. Resource fields may differ between result JSONs without changing scientific tensors or comparison metrics. Both previous failed prefix attempts and their reservations remain failed and charged.

## Supported executable delivery

Candidate: `minimind_re.softmax_bucket_candidate.SoftmaxBucketCandidate`, source[softmax_bucket_candidate.py](../src/minimind_re/softmax_bucket_candidate.py), SHA256`b9d685f1af23267b7e8c904b01a06e81ae9d7e47fdf71a42688b2b55000c88b3`. Independent reference context: `minimind_re.softmax_bucket_reference.softmax_bucket_reference`, SHA256`fb8ec4b8218ac7e49087f5dfb44bee535681696e3049678a83b631f10b43d464`, applied to `oracle.adapter.ReferenceAdapter`. Candidate does not call the reference forward.

Use the frozen checkpoint SHA256`5b8c49f6c9d965092e651cafeeaeb8705558632b3fd4ac8ab319cbc5a3cbc4a0` and[supplied_config.json](../artifacts/allowed/supplied_config.json), CPU FP32 torch2.11.0+cu128, two threads, deterministic algorithms, MKL_CBWR=AUTO,STRICT and MKL_DYNAMIC=FALSE. The tested interface is batchone, int64tokens/positions, boolean complete attention mask, genuine eight-layer KV tuples and single-token cached append. This is a pinned local mode, not an unrestricted model API compatibility claim.

The executable full recipe is `python -m minimind_re.softmax_bucket_matrix_probe --packet <frozen-packet>` under its reviewed guard, never a standalone bypass. [Launch intent](../evidence/cache-softmax-matrix/launch-intent-v1.json) records exact cwd/guard/policy/admission/role commands. Existing run IDs and outputs are spent and must not be rerun. A reproduction needs new immutable packet/output identities, retained ledger accounting and exact independent admission; do not set the admission environment variable manually.

Inspect existing evidence without a model load:

```bash
/usr/bin/python - <<'PYCODE'
import json
from pathlib import Path
root = Path('EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-cache-softmax-matrix-001-20261118')
for role in ('producer', 'independent'):
    path = root / 'runs' / ('cache-softmax-matrix-' + role + '-v1') / 'result.json'
    if path.exists():
        result = json.loads(path.read_text())
        print(role, result['status'], len(result['rows']), sum(not row['pass'] for row in result['rows']))
PYCODE
```

## Resources and remaining boundary

Both roles reserve50,600 each, retaining1,138,656previous charges; final charged lifetime 1,239,856. CPU2,10GiBownedRSS,900secondsperrole/3600phase,4GiBphase disk,12GiBcombineddisk and2GiBfree reserve are explicit reviewed limits. Every failed prior reservation remains charged. Producer conservative RSS1,382,518,784bytes passed. Independent accounting confirms 1,239,856 charged, zero active runs and phase size 4,035,646,419 bytes below 4 GiB; combined disk/free reserve checks passed.

This report changes the supported numerical mode, not historical facts. It leaves original native R2FAIL and rejected-mode fullmatrixFAIL visible. It does not prove semantic task accuracy, unique source recovery, a complete circuit or universal numerical equivalence. The91tensor map, actual-input trace and separately verified16/16held-out echo intervention supply distinct evidence; see the[workbench](reverse-engineering-workbench.md). Whole-objective R0–R4/B1–B6 integration remains for separate review.
