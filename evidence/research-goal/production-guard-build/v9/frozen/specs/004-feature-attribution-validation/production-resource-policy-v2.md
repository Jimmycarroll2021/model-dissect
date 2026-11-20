# Prospective production resource policy v2 — storage-only amendment

Status: USER_APPROVED_PROSPECTIVE_STORAGE_POLICY; exact artifact pending independent review. This is not runtime admission or a claim of production feasibility. Authority is the user's explicit approval communicated in the coordinator assignment: 29 GiB retained evidence PER ROLE and 96 GiB TOTAL project reservation under `/mnt/d/AI`.

This new prospective version references immutable `selected-claim-v1.md`, SHA256 b3500849251ffd2e7acb93fa58377f735fe3c5d5466796317bb98e06e58f5590, specifically its resource paragraph at line49. That file, corpus, freeze manifest and historical receipts are unchanged. Its historical 1 GiB output cap is preserved in v1; this v2 changes only the prospective retained-evidence ceiling and destination/project storage envelope. No previously failed or over-budget attempt is retroactively admitted.

## Approved storage terms

- Retained evidence cap per independent role: **29 GiB = 31,138,512,896 bytes**. All main scientific arrays, supplements, metadata, controls, gates, entry/completion records, reports, retained partials and failures share that role's aggregate cap. No per-child multiplication, hidden extra archive, deduplication assumption or free retry.
- Total project reservation envelope: **96 GiB = 103,079,215,104 bytes**, under new exclusively owned subtrees of **`/mnt/d/AI`**. This is the total for both roles and shared assets, not 96 GiB per role. It includes retained evidence, required original artifacts, separate derived caches, staging/partials and allocated-space overhead. This policy establishes a planning envelope; no actual allocation or reservation has been performed.
- Preserve and account for previous failures and retained partials; charge retained destination bytes against the applicable aggregate envelope. Preserve earlier attempt ledgers and their call/time/position charges. No reset, uncharged copy, deletion, move, or repeat of an expensive attempt is authorized by this amendment. Existing originals remain in place.

## Unchanged scientific and execution contract

All scientific claims, feature identity, corpus, control construction, numerical tolerances, eligibility, replacement gates, thresholds, bootstrap seed/resampling, independent-repeat requirements and verdict rules from v1 remain unchanged. No numerical criterion is relaxed.

Per-role caps remain **1,024 calls**, **65,536 input positions**, **1,800 seconds wall time**, **24 GiB owned RAM = 25,769,803,776 bytes**. CPU policy is unchanged. Maximum input64tokens, batch1, no cache remain unchanged. Zero new hosted spend remains binding. This amendment supplies no runtime, network, model-download or checkpoint-access admission. Authorized locally entitled original checkpoint access remains pending; actual target/scientific status remains INCOMPLETE.

## Conditional accounting and implementation prerequisites

The independently reviewed proposal in `evidence/research-goal/production-resource-proposal/v1` and `production-resource-proposal-review/v1` supports a conditional per-role envelope30,321,607,968bytes, rounded up to29GiB. It includes a proposed hard1GiB allowance for production ledger/report/input publications that is not yet an implemented, verified bound. Two role caps plus shared model tensor payload/original dictionaries and separate derived caches total96,332,282,880bytes, leaving6,746,932,224bytes within96GiB for serialization/tokenizer/config, allocation overhead, staging and partials. This is not a guaranteed fit. No extra full model copy is assumed.

Before execution: implement/review a shared aggregate reservation and partial-charge ledger, concrete production extras/publication limits, exact destination path guards, resource binding and external guard ceilings consistently with this version. Preserve per-array/metadata bounds unless separately prospectively reviewed. Inventory actual assets and allocated bytes, include all retained attempts, and recheck free space immediately before staging. Authenticate actual original model/tokenizer, reconcile complete source pins, and independently admit a concrete combined candidate with CPU/RAM/time feasibility. This document does not change executable enforcement; older code ceilings remain until an independently reviewed implementation updates them.

## Bounded filesystem observation

Evidence lives in `evidence/research-goal/production-resource-policy-v2/preflight-result.json`. A unique owned fixture under `/mnt/d/AI` tested canonical non-symlink ancestors, exclusive creation, O_NOFOLLOW rejection, stable directory/file inode identities, file/directory/parent fsync, same-directory rename and exact-byte replay, with a retained partial. The successful calls establish only observed filesystem API behavior. Fsync return is not proof of crash or power-loss recovery, and this is not a hostile concurrent mutation test. Capacity is a dated snapshot, not reserved space. No96GiB allocation, model acquisition, source move or deletion occurred.
