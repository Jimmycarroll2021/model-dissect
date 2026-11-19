# ECHO-MECHANISM-001 frozen protocol

Status: candidate, NOT_RUN. This is source-assisted instrumentation of spontaneous digit echo, not task-answer accuracy, repetition/EOS explanation, necessity, uniqueness, a complete circuit or source-position attribution. Native reference creates unused cache outputs; every invocation has cache=None and no cache is reused. CPU FP32/eval, deterministic algorithms, explicit attention, two threads, same checkpoint/tokenizer/config. No training, downloads, installs or GPU.

## Fixed inputs and qualification

`fixtures-v1.json` contains exactly 8 development and16 held-out pairs,48 distinct member texts/derived token arrays. It pins34 exclusion sources covering all existing fixture JSON and atlas/behavior/continuation raw result JSON. Every nested string is excluded. Each pair has four distinct digits a,a',b,c; two members change only a. Start with lexicographic permutations, shuffle with Python Random(172903), then sequentially minimize `(max(exposure[a]+1,exposure[a']+1),exposure[a]+exposure[a'],shuffle_rank)` among tuples whose two member texts are unseen. First8 are dev, remaining16 held. Exposures4–5 per digit across48 members. No reversed/reused members. This replaces the ambiguous first-shuffle/balance proposal with one deterministic algorithm. Exact fixtures are published before any model call.

Tokenizer validation is inside admission before loading weights. Every prompt must equal its frozen48 IDs, digit labels are exactly51+digit, sole changed token index8; prefix `甲写下数字` must equal [5078,1226,504,1372]. Any mismatch stops FIXTURE_UNAVAILABLE; no replacements. Each member actually generates five greedy tokens (full prefixes48..52). Both natural outputs must be prefix+correctdigit, and common-prefix baseline D=z[a]-z[a'] must be >=2 clean and <=-2 corrupt. Common-prefix intervention on an ineligible member is only a forced-prefix diagnostic. Fixed denominators8/16; eligibility does not remove rows. Nonfinite logits stop the entire experiment rather than silently dropping a row.

## Exact instrumentation/parity schedule and resource accounting

Each role reserves78112 positions, combined156224, not refunded. Actual full-path maximum76632 positions/1480 forwards:
- Native and reconstructed full-boundary parity at52 and128:360 positions/4 calls.128 is cyclic repetition of dev00 clean prompt+prefix, truncated. All embedding/block/final/logit tensors: everyelement absdiff<=1e-5+1e-4*abs(reference), finiteFP32, exact all-position logits argmax. Hidden argmax descriptive only. Save native tensors and per-boundary hashes/errors. Models sequential, delete/gc before next.
- Native plain52, all8 observers52, simultaneous self-copy of all64 head slices52, hook-removed restored52:208/4 calls. All full-vocabulary final-position logits exact. Per-layer hook fires exactlyonce; everyhead self-copy assigns its96-value slice in a clone, asserts untouched complement exact, validshape/device/dtype. Missing/failinghooks stop before any search. Exception teardown tested synthetically. This is simultaneous all-head preservation, not64 individually observed noop forwards.
-24 pairs ×2 ×sum(48..52)=12000/240 natural qualification.
-24×2×52=2496/48 activation baselines.
-8×64×2×52=53248/1024 development head patches.
-16×10×52=8320/160 held arms:2baseline repeats,2target patches,2control patches,2combined target+control self-copy noops,2restorations. Combined self-copy tests both selected heads in each of2forwards; each selected slice asserted separately. Individual target and control noop forwards are not claimed.

Total core76064+parity568=76632.2048/role reserved parity allowance leaves1480 unused/role. No other forwards permitted, no autoexpansion/retry. Each call charged before execution, records actual length, device/dtype, full6400-vector SHA, top1 and raw finalvector file. Every repeated call label/length/hash/top1/accounting is compared exactly, ignoring only differing output filepaths. Repeat also compares prepared fixtures, all boundary parity metrics/hashes, full headscan scores/selection/held results. Both roles repeat the same conditional stops.

Historical four ledgers bind original175600+atlas120000+behavior155328+continuation353280=804208. All source/snapshot SHA and zero-active verified, all four prior lockfiles held read-only. No oldclock/ledger reset. New phase cap3600seconds starts only admission;900seconds producer/900repeat, independent reserve retained,8GiB ownedRSS,512MiB phase disk,8GiB combined original+fourphases,2GiB free reserve. Lifetime ceiling960432 after two reservations,39568 remains below1000000. No phaseclock exists at candidate freeze.

## Intervention, scoring and selection

Native path `adapter.model.model.layers[l].self_attn.o_proj` input is concatenated8×96 heads. A forward prehook clones input `[1,52,768]`, replaces exactly `[0,51,96*h:96*(h+1)]` from matched opposite member's baseline vector, asserts every othercoordinate unchanged, and returns tuple(clone). No parameters change; complete in-memory state hashes checked before/after and checkpoint identity remains bound. All handles removed in finally, preexisting/leakedhooks refuse execution. Both directions: clean source into corrupt destination, then corrupt source into clean.

Rf=(Dpatched_corrupt-Dcorrupt)/(Dclean-Dcorrupt); Rr=(Dclean-Dpatched_clean)/(Dclean-Dcorrupt). Raw ratios unbounded. Undefined nonpositive/nonfinite denominator -> null with reason. Any natural/margin/denominator failure masks BOTH target/control primary min(Rf,Rr) to0 and positive movement false; paired advantage0. Raw observed deltas retained, no filtered denominators.

Dev eligibility>=7/8 or stop. Scan all64 heads and bothdirections for all8. Target maximizes median primary minimum recovery; ties ascendinglayer/head. Require median>=.25 and positivebidirectional>=6/8 or stop. Same-layer control excludes target and minimizes abs(median primary minimumrecovery), tieascendinghead. This deliberately weakest development-selected control is not random/population evidence. Freeze selection receipt before heldout, no reselection.

Held success: eligibility>=14/16 AND bidirectionalpositive>=13/16 AND median primaryminrecovery>=.25 AND median pairwise(targetprimary-controlprimary)>=.15. All16 retained. Noop/restoration exact full6400vector, any mismatch invalidates experiment. Descriptive sign statistic is count of strictlypositive paired advantages over fixed16 under Binomial(16,.5), one-sided P(X>=observed). Zeros/ineligible are non-successes. Common template/dependence and dev-selectedcontrol prevent treating this as an independent random-sample proof.

Save all64 head clean/corrupt/difference/projected-difference norms for each baseline; held selected/control full96vectors and768 projectedcontributions use existingcapturedactivations and W_O, no extraforward. These traces do not demonstrate attention to a sourceposition. Dynamic intervention, not staticnorm, supports any bounded causal conclusion.

Stop on resource/identity/parity/hook/finite/stability/eligibility/selection/heldfailure. Preserve incremental journals. No alternate prompt, MLP, position or length search. Independent protocol admission required before actual execution; broader reverseengineering goal remains active.
