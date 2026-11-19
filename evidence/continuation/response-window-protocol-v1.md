# CONTINUATION-001 v1 — four fixed response-window diagnostics

Candidate for independent review, no real model/tokenizer execution yet. The user explicitly renewed the broader goal; this bounded step measures response-window adequacy. Spec Kit remains authoritative; long-horizon/autoresearch principles structure evidence iteration without a second workflow controller. TASK_STATE/CASE_FACTS have a separate assigned writer. Historical findings and ledgers are immutable.

## Frozen question and outcome

On four source-selected inputs, does unchanged greedy/non-thinking computation reach EOS within256 generated tokens, and what answers does it actually produce? This extends the24-token observation window only; no template, weights, sampling, dtype, task wording or cache changes. Three reused diagnostic cases are not a held-out family benchmark. Completion alone is not correct mathematics or a mechanism. Source research: evidence/inference-diagnosis/research.md. The negative24-token screen and cache/full R2 FAIL are preserved.

Inputs in fixed order: existing behavior-copy-screen-00, behavior-subtract-screen-00, behavior-compare-screen-00, plus exact upstream first evaluation question `你有什么特长？`. Original task prompt IDs/rendered bytes stay identical. Runtime tokenizer preparation uses pinned local API with one user message, add_generation_prompt=true, open_thinking=false; records all input IDs, repr/rendered hashes before weights load. The three old lengths are48,30,30; new fourth input must have1–64tokens or abort, no substitutions. Prepared JSON must repeat exactly independently. No preparation tokenizer runs outside admitted model processes.

## Charged longer-prefix fidelity check

Before generating any response, compare unchanged ReconstructedAdapter against unchanged evaluator ReferenceAdapter at lengths128,256,319. Construct each deterministic unpadded prefix by repeating the exact COPY screen00 input token list and truncating to length. This synthetically exercises extended positions; it is not meaningful-language behavior evidence.

CPU FP32/two threads/deterministic operations/cache=None. Each of3 reference forwards and3 candidate forwards is charged by actual input length:2*(128+256+319)=1406positions/run, reserved2048/run. Retain entire reference and candidate embedding, block0, final-normalized and logit tensors in separate .pt files and per-check event receipts. Require every element of all four boundaries finite, matching shape/dtype, every element abs(candidate-reference)<=1e-5+1e-4*abs(reference), and exact full-vocabulary logit argmax at EVERY position. Hidden-boundary argmax is descriptive only, not a new gate. Report maximum absolute error and tolerance ratio. Any failure prevents all generation and remains a charged technical failure. No comparator relaxation or retry. Reference model is deleted before candidate load to reduce simultaneous memory; source is used only to validate the independently composed candidate.

This small checked set is explicitly NOT full-model equivalence or a repair of cache/full fidelity. Do not extrapolate exact equality to all prompts/prefixes/backends.

## Generation and preservation

Generate using unchanged reconstructed forward, CPU FP32/two threads, every call full-prefix/cache off. Max256 newly generated tokens INCLUDING EOS-producing token; only pinned EOS2 stops early. Max prompt64 implies max forward prefix319. Full6400-vocabulary greedy argmax; lowest token ID on ties; no suppression, prefill, sampling or hidden controls. EOS at token256 is complete; no EOS by256 is censored/incomplete. No automatic expansion.

Record each token ID, chosen logit, full-vocabulary vector SHA256, input length, EOS probability and EOS rank. Rank uses descending logits/lowest-ID ties; probability uses FP64 softmax for reporting only, never changes token choice. Persist after every step/case with fsync. Decode body with skip_special_tokens=false and clean_up_tokenization_spaces=false; exclude only the final EOS and retain any other controls0–35. Controls are reported, never hidden or reinterpreted as correct content. Actual generation positions for n calls and prompt L: C(L,n)=nL+n(n−1)/2, including EOS-producing call.

First24 token IDs, chosen logits, full-vocabulary hashes and recorded step/input lengths for the3 reused prompts must exactly match previous response-screen-v1.json. Drift is technical failure; preserve raw observations and stop this process rather than reinterpret a changed baseline. The fourth prompt has no old continuation baseline.

## Termination and frozen answer rubric

Termination is the primary diagnostic metric. Report EOS/cap, token count and full response separately from meaning. The old conservative parser is an unchanged secondary comparability field, not the definition of semantic correctness of an explanation.

For each task response, two separately operated graders apply this rubric to the frozen full text, question, expected value/relation and control/EOS record; neither may change the grammar or prompt. Raw responses are immutable. Reconcile agreement; unresolved grading disagreement => AMBIGUOUS, not a favorable vote. Preserve both assessments and rationale.

1. No EOS => INCOMPLETE. Non-EOS control tokens => CONTROL_CONTAMINATED; do not hide them for a correctness verdict.
2. A complete ordinary response is CORRECT only if it explicitly gives the unique expected answer and contains no unretracted contradictory answer or false mathematical assertion affecting that answer.
3. Explicit incorrect final value/relation => WRONG. An explicit false intermediate mathematical claim counts WRONG unless clearly identified and corrected/retracted before a unique correct conclusion. A numerically correct final answer does not erase unretracted contradictions.
4. Multiple competing answers without an explicit final resolution, no explicit answer, or unclear reference/relationship => AMBIGUOUS. An explicit correction/retraction can resolve an earlier error only if it clearly disowns it; record that repair in rationale.
5. General-chat control gets termination and COHERENT_ASSISTANT_RESPONSE / INCOHERENT / AMBIGUOUS only; it has no numeric expected answer. Coherent means intelligible and relevant self-description of capabilities; it does not verify claimed capabilities. Control contamination/incompleteness take precedence.

No percentage qualification claim from three reused task questions. If only chat terminates, distinguish task-specific difficulty. If arithmetic terminates incorrectly, preserve that observation. No conditional intervention is enabled here. After the fixed4 cases and independent repeat, report every outcome and choose any later scientific protocol separately under the active broad goal.

## New resource boundary and reproducibility

Two immutable IDs only: continuation-producer-v1 and continuation-independent-v1. Each reserves174592generationpositions (125568for the3known lengths,49024for fourth cap64),2048paritypositions, total176640. Actual parity consumes1406; unused reservation retained. Combined353280 new reservations plus historical450928 gives804208 lifetime maximum,195792 below1000000. No additional slots/refunds/retries. Tokenizer work is inside charged wall time; no model forwards are hidden in preparation.

Each process cap900seconds; total1800modelseconds. New separate phase deadline3600seconds from first explicit reviewed admission. Producer must leave900seconds and176640positions for independent repeat. New source/runtime workspace EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-continuation-001-20261118. All three source ledgers and snapshots (original175600, atlas120000, behavior155328) must hash-match and show zero active rows, and all three source locks are held throughout runtime. Carry450928 forward; never change old clocks/ledgers. Fixed new policy hash binds new ledger; no resetting a partially used phase.

Guard8GiB owned RSS;512MiB new phase disk;8GiB combined original+atlas+behavior+new directories;2GiB free reserve. CPU-only2threads, no installs/downloads/training/spend/publication. Guard checks are process supervision, not OS isolation. Long-prefix memory acceptance requires live measured guard success; previous shorter-prefix memory is not proof. Estimated tensors for6 saved parity arrays/run are~47MiB (1406*(6400+3*768)*4bytes),~94MiB both runs, within staged disk allowance. Reserve process failure remains charged. Dependencies and original reference/candidate/config/tokenizer identities must be bound in exact independent admission.

Independent repetition runs its own charged3-prefix parity and same4responses. Require exact preparation, token/step/hash/stop/control/parser records and raw journal bytes; separately check all parity comparisons against saved arrays. Independent repeat binds exact producer output/preparation/receipt. No model/tokenizer load is allowed by this document alone; no implementer-created PASS admission.

Acceptance: D1 frozen protocol/fixtures/rubric/resource/source review; D2 actual long-prefix parity and4responses or honestly preserved technical failure; D3 independent exact repeat and separate grading; D4 plain-language report with all responses, censoring/correctness distinction, exact artifact/resource receipts and next scientific decision. This step does not by itself complete the broader goal.
