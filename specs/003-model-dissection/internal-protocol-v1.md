# TRANSFER-003 scientific protocol v1 — bidirectional head transfer

Status: PREREGISTERED DESIGN FOR INDEPENDENT REVIEW; no tokenizer/model load or scientific execution. Exact text fixtures below are frozen. Token arrays, implementations and packet admission must be frozen independently before execution. This document does not authorize target execution.

## Question, scope and separation

For one development-selected GPT-2 preprojection head, does swapping the clean/corrupt final-token head vector transfer a baseline-derived output contrast more than a norm-matched orthogonal change? This is distributional mediation on these prompts, not semantic correctness, head necessity or a complete circuit. Positive and adequately controlled negative findings both complete the bounded investigation. Inadequate eligible data or invalid controls yield INCOMPLETE, never REFUTED.

Identity: openai-community/gpt2 revision607a30d783dfa663caf39e06633721c8d4cfcd7e; original checkpoint/config/tokenizer and native/candidate source hashes must enter admitted manifest. CPU FP32 eager inference/eval, two threads, no cache, no GPU. Native/candidate both execute every arm with identical masks/positions and algorithmic edits; candidate baseline parity alone is insufficient.

## Frozen fixtures and admission

Use exactly the24 pairs below:8 development and16 held-out. Tokenize with pinned tokenizer, add_special_tokens=False; batch size1 and NO padding for this protocol. Require 2<=length<=32, equal clean/corrupt lengths, exactly one differing input token, identical suffix after that token and identical final token ID. Positions0..S-1, all-one mask; edit/readout at S-1. Record full token arrays and first differing index. Tokenization failure is an explicit fixture-admission failure; no automatic prompt replacement. A revised corpus requires new protocol version before any corresponding held-out run. All24 text fixtures are disclosed for audit; held-out means excluded from execution-based site selection, not secret text. Executor must not inspect held-out model outputs before selection lock.

- dev-01: clean `The code is red. The code is`; corrupt `The code is blue. The code is`.
- dev-02: clean `The key is green. The key is`; corrupt `The key is black. The key is`.
- dev-03: clean `The flag is white. The flag is`; corrupt `The flag is brown. The flag is`.
- dev-04: clean `The sign is pink. The sign is`; corrupt `The sign is gold. The sign is`.
- dev-05: clean `The box is blue. The box is`; corrupt `The box is red. The box is`.
- dev-06: clean `The bag is black. The bag is`; corrupt `The bag is green. The bag is`.
- dev-07: clean `The hat is brown. The hat is`; corrupt `The hat is white. The hat is`.
- dev-08: clean `The cup is gold. The cup is`; corrupt `The cup is pink. The cup is`.
- held-01: clean `The door is red. The door is`; corrupt `The door is blue. The door is`.
- held-02: clean `The wall is green. The wall is`; corrupt `The wall is black. The wall is`.
- held-03: clean `The car is white. The car is`; corrupt `The car is brown. The car is`.
- held-04: clean `The boat is pink. The boat is`; corrupt `The boat is gold. The boat is`.
- held-05: clean `The book is blue. The book is`; corrupt `The book is red. The book is`.
- held-06: clean `The shirt is black. The shirt is`; corrupt `The shirt is green. The shirt is`.
- held-07: clean `The coat is brown. The coat is`; corrupt `The coat is white. The coat is`.
- held-08: clean `The shoe is gold. The shoe is`; corrupt `The shoe is pink. The shoe is`.
- held-09: clean `The ball is red. The ball is`; corrupt `The ball is blue. The ball is`.
- held-10: clean `The card is green. The card is`; corrupt `The card is black. The card is`.
- held-11: clean `The pen is white. The pen is`; corrupt `The pen is brown. The pen is`.
- held-12: clean `The chair is pink. The chair is`; corrupt `The chair is gold. The chair is`.
- held-13: clean `The desk is blue. The desk is`; corrupt `The desk is red. The desk is`.
- held-14: clean `The room is black. The room is`; corrupt `The room is green. The room is`.
- held-15: clean `The house is brown. The house is`; corrupt `The house is white. The house is`.
- held-16: clean `The bus is gold. The bus is`; corrupt `The bus is pink. The bus is`.

## Baselines and eligibility

For each pair run native and candidate clean/corrupt once each, capturing final-real-position full50257 logits and all heads at preprojection layers{2,5,8,11}. Finite FP32 outputs and native/candidate elementwise parity abs(a-b)<=1e-5+1e-4*abs(native), exact argmax are mandatory. Metric calculations use float64 converted from saved FP32 arrays, never additional forwards.

From native baselines only compute delta_v=clean_logit[v]-corrupt_logit[v], vocabulary indices0..50256 inclusive. Target t is argmax(delta_v), distractor d is argmin(delta_v), smallest vocabulary index breaks ties. Define D(x)=x[t]-x[d], denominator A=D(clean)-D(corrupt). Baseline eligibility requires t!=d and finite A>=0.5. No requirement of correct language answers or clean/corrupt zero crossing is claimed. This deterministic baseline-derived contrast is fixed before any intervention for that pair; neither intervention outcomes nor candidate outputs select t/d.

Site eligibility also requires each actor's selected clean/corrupt64-vector difference norm>=1e-6 and valid orthogonal control in both directions. Development site qualifies only if all8 preregistered pairs meet baseline/site eligibility; held-out selected site requires all16 pairs eligible. Every exclusion remains reported. Missing eligibility yields INCOMPLETE; do not choose only favorable eligible cases or alter denominators.

## Head seam and exact edits

Native seam transformer.h[L].attn.c_proj INPUT [1,S,768], validated as [1,S,12,64]. Candidate equivalent preprojection seam; preserve original packed output projection arithmetic. Require exactly one target hook invocation per call. Before target admission independently reconstruct attention-weighted-V for one trace and compare its reshaped values with the native captured c_proj input using unchanged tolerance. This checks head layout, not only reshape dimensions.

Sites in lexicographic order: Cartesian product layers{2,5,8,11} and heads{0,4,8}, exactly12. At each pair/site run BOTH directions: corrupt recipient<-clean donor; clean recipient<-corrupt donor. Donor always comes from that actor's immutable baseline capture, never a previous edited run. Replace only64 coordinates at last real token and selected head; all complement coordinates must remain bitwise unchanged. Save full pre/post selected-layer arrays, complete donor/recipient/edited64-vectors, and elementwise complement verdict/hash. Original weights/config/tokenizer/source identities unchanged before/after; hook teardown must run on exceptions.

Orthogonal control: float64 delta=donor-recipient, n=norm(delta). Let u=roll(delta,1); remove projection onto delta: u=u-delta*dot(u,delta)/n². If norm(u)<1e-12*n, choose basis e_k with smallest abs(delta[k]) (lowest index tie), then u=e_k-delta*dot(e_k,delta)/n². Set c=n*u/norm(u); write float32(recipient+c). Check finite c, abs(dot(c,delta))<=1e-10*n² and abs(norm(c)-n)<=1e-10*n in float64 construction. After FP32 casting check effective perturbation norm error<=1e-5+1e-4*n and absolute normalized dot<=1e-4 (denominators n*norm(effective_c)); otherwise control invalid/INCOMPLETE. Target writes exact FP32 donor. This direction-specific deterministic construction uses no random draws and applies independently to each actor's baseline vectors.

No-op returns recipient unchanged; restoration is a fresh original-recipient call after successful teardown of real patch/control hooks. No-op and restoration saved final logits and selected-layer activation arrays must be bitwise equal to that actor's baseline. No weight surgery/training. Both actors execute target, control, no-op and restoration; matched resulting final vocabulary logits and head arrays pass fixed tolerance and exact argmax, including actual edited arms. A parity/control failure blocks scientific acceptance regardless of effect size.

## Development selection and held-out lock

Development:8 pairs x12 sites, both directions, target/control arms, both actors. No-op/restoration run on all8 development pairs in both directions/both actors at the SELECTED site after selection; synthetic tests separately exercise all12 hook locations before admission. Selection uses native metrics only: per direction recovery R=(D(edited)-D(recipient))/(D(donor)-D(recipient)); orthogonal recovery C analogously; pair primary P=(R_forward+R_reverse)/2 and pair advantage Q=((R_forward-C_forward)+(R_reverse-C_reverse))/2. No clipping, absolute value or sign reversal.

Select qualifying site with highest median Q over all8 development pairs; exact tie chooses lexicographically smallest(layer,head). If no site qualifies, INCOMPLETE. Save immutable selection receipt with all12 site metrics, eligible flags, selected site, hashes and timestamp before first held-out forward. Verifier recomputes selection. No candidate-informed or held-out-informed reselection. Independent repeat replays full development schedule and must recover same selected site, then all held-out arms.

## Fixed scientific verdict

Primary held-out inference uses native pair P and Q; candidate reproduces the same science as a fidelity gate, not a second independent population. SUPPORTED if all16 held pairs eligible, all numerical/control/restoration gates pass, median(P)>=0.20, median(Q)>=0.10, and at least12/16 pairs have Q>0. Otherwise REFUTED for this exact bounded threshold claim if all16 eligible and all validity gates pass. Invalid controls/parity/insufficient cases yield INCOMPLETE. Thresholds are engineering preregistration choices, not claims derived from papers or universal effect sizes. Candidate per-pair P/Q must be recomputed and absolute differences <=0.002; mismatch is scientific reproducibility failure. Independent repeat raw identity/metrics must match under frozen environment; tolerance is not license to change source or input.

## Secondary metrics from saved logits only

Report directional raw gap shift D(edited)-D(recipient), target-minus-control gap shift, and donor-directed KL improvement KL(p_donor||p_recipient)-KL(p_donor||p_edited) for target and orthogonal arms. Compute float64 stable log_softmax over complete50257 saved final-position logits; p_donor=exp(logp_donor), KL=sum p_donor*(logp_donor-logp_other). Record underflow counts and fail nonfinite metrics; do not silently drop cases. Baseline metrics never imply correctness of target token.

Paired bootstrap: NumPy Generator(PCG64(20261119)),10000 resamples of16 pair indices with replacement, both directions and arms retained together; recompute median(P), median(Q), and mean pair-averaged target-minus-control KL improvement. Report 2.5/97.5 percentiles using linear interpolation. Bootstrap intervals and secondary metrics are descriptive and cannot override primary verdict. Record NumPy version and indices or their deterministic hash; no extra model calls.

## Exact call accounting per producer or independent-repeat role

Development baselines8*2recipients*2actors=32 calls. Development target/control8*12sites*2directions*2arms*2actors=768 calls. Selected-site development no-op/restoration8*2directions*2arms*2actors=64 calls. Development subtotal864.
Held-out baselines16*2recipients*2actors=64 calls. Held target/control16*2directions*2arms*2actors=128 calls. Held no-op/restoration16*2directions*2arms*2actors=128 calls. Held subtotal320.
One additional actual walkthrough input (dev-01 clean) run once per actor=2 calls; collect all15 core boundaries and all12 layers' attention/head observations needed for map/trace. Total1186 calls per role; two roles2372 calls. Head captures are in those calls and require no extra forward. Layout recomputation is from captured Q/K/V/attention using primitive operations and must be separately timed/counted as primitive work. No hidden warmup, token generation or probing call.

Maximum1186*32=37952 physical model-token positions per role; exact token-frozen schedule replaces maxima before admission. Reserve40000 positions,1200 seconds and1 GiB raw/output per role, inclusive of all failures/retries and trace. Both roles80000 reserved positions/2400 seconds/2 GiB. These fit initial science ceiling100000 positions/2400seconds/8GiB but are conditional on measured timing pilot and remaining global ledger; no extension if predicted time exceeds cap. The separately admitted small numerical timing pilot has its own charge. No scientific target run before runtime/science implementation review and selection-lock enforcement are tested.

## Minimal sufficient lossless retained evidence

For every call retain complete50257 FP32 final-real-position logits, exact input/mask/positions, actor/arm/site/direction and donor linkage. This intentionally retains final readout only for science; full sequence/all-boundary fidelity is covered by the separate numerical matrix. Baselines retain all four searched layers' full preprojection arrays. Actual edits retain full selected-layer pre/post arrays and64-vectors; no-op/restoration retain selected-layer arrays and compare with retained baseline. Trace retains all15 boundaries at every physical row for both actors plus per-layer Q/K/V, attention probabilities and all-head preprojection values for one actual input. Its static map links each operation to atlas weight keys/Conv1D orientation/tied embeddings and labels source-assisted versus observed evidence.

Conservative main-array estimate below600 MiB/role at32 positions: final logits about228MiB; edited pre/post arrays168MiB; baseline four-layer arrays36MiB; no-op/restoration pre/post arrays36MiB; trace boundary/QKV/attention arrays under32MiB, plus headers/configs/receipts. Enforce1GiB total packet output, measure real serialization before execution, and never silently omit mandatory arrays to fit cap. Content-addressed deduplication allowed only with exact logical reconstruction. Independent verifier repeats1186 calls, recomputes all masks/contrasts/control norms/complement checks/parity/metrics/bootstrap/selection and artifact preservation from actual retained evidence. Source-assisted bounded evidence only.
