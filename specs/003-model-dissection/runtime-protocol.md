# TRANSFER-003 initial resource protocol v2

Status: PROPOSED_FOR_INDEPENDENT_REVIEW; build/synthetic only, target runtime NOT_ADMITTED.

## Fixed ceilings and phase identity

New phase workspace proposed: EXTERNAL_ARTIFACT_ROOT/workspaces/model-transfer-003-gpt2. New ledger under phase/ledger.jsonl; never reuse Pythia phase start/ledger. Start clock only on first admitted target/static packet, record immutable start and identity. Historical combined reservations 1299306 positions remain closed and unchanged; proposed third ceiling 250000 new reserved positions gives prospective combined ceiling 1549306, not a claim of consumption.
CPU only, 2 threads, no GPU; inference/eval FP32 eager. Owned memory cap 10 GiB; minimum available RAM 2 GiB; stop admission if ownership/reserve cannot be established. No swap growth assumed as capacity. Added phase disk ceiling 32 GiB including acquired files, synthetic outputs, raw evidence and failed packets; maintain at least 2 GiB free on EXTERNAL_ARTIFACT_ROOT. All bulk/synthetic outputs and caches use this workspace, never root filesystem. Initial acquisition ceiling 600 MiB; original bundle expected 550967953 bytes. No cleanup of prior data.
Aggregate reserved process time ceiling 4800 seconds; phase wall ceiling 7200 seconds; per-call limit 120 seconds or remaining packet time, whichever smaller. One target process at a time; independent repeat runs serially. Reserve at least 40% unspent positions/time/disk for independent repeat until exact schedule replaces that allowance. Reserve before launching; keep failed reservation charges; record actual calls/physical positions/bytes/time. No automatic phase resets.

## Staged packets

1. Acquisition after independent selection PASS; verify bytes/hashes without loading target model. Static atlas after artifact audit and exact build review: 0 forward positions, at most 180 seconds and 256 MiB output; every tensor statistics required, bounded optional spectrum analysis below.
2. Producer pilot: at most 24 calls/156 physical positions, reserve 256 positions/120 seconds/128 MiB raw output. Same matrix arms and boundaries as numerical protocol, four small fixtures. Load, real serialization/reload, digest and guard checks count in elapsed budget. If any gate fails or projection exceeds remaining phase caps, stop and retain failure. No full matrix automatic launch.
3. Independent pilot/review: reserve 256 positions/120 seconds/128 MiB; verify measured feasibility and serialization. Numerical full producer/repeat each provisional reserve 2800 positions/600 seconds/896 MiB. Exact runtime may require revised schedule within ceilings; independently review before execution, never silently extend.
4. Science discovery and held-out schedules require separate frozen packet budgets derived from pilot. Combined science maximum proposed reservation 100000 positions/2400 seconds/8 GiB, inclusive of independent repetition and failed retries. This is a ceiling, not authority to spend all of it. Remaining phase capacity is contingency requiring explicit technical admission, not an invitation for more search.

## Static analysis limits

Account every stored tensor and finite/min/max/mean/std/norm statistic; distinguish parameters, aliases/ties and causal buffers. At most 12 bounded spectrum analyses on deterministic submatrices no larger than 256x256, explicitly labelled samples rather than full matrix singular spectra. At most 48 static plot/views. No all-head H-squared or vocabulary matrix exports; weight originals are already retained losslessly. Static output timeout does not permit silently omitting mandatory tensor coverage.

## Readiness and enforcement

Refresh free disk/RAM, running target ownership, CPU load and relevant process state immediately before admission; other workloads untouched. Reuse existing phase guard in clean directory first; measure overhead before changes. Guard must fail closed on missing ledger, unresolved prior reservation, over-cap packet, lost free reserve and serialization failures; independently test cancellation/failure/reconciliation using synthetic data. Proposed limits are not proof of implemented enforcement. No target run until exact guard/oracle/candidate/protocol review PASS.

## Effective guard versus design ceiling (review correction v3)

Coordinator reports inherited/current guard design enforces24GiB disk and3600 aggregate process seconds, below proposed32GiB/4800seconds. Until separately reviewed implementation changes, effective limits remain24GiB/3600seconds; no packet may rely on larger document-only caps. Source caps must bind250000 new positions plus1299306 historical positions =1549306 prospective combined ceiling. Neither historical reservations nor the new wall clock may be reset.

Current planned runtime reservation arithmetic: static producer/repeat180seconds each=360; two calibration pilots120seconds each=240; full core producer/repeat600seconds each=1200; scientific producer/repeat1200seconds each=2400. Sum4200seconds exceeds effective3600seconds (but fits proposed4800seconds). Therefore all full repeats DO NOT yet fit the inherited guard and science admission is BLOCKED until pilot measurement supports a separately preregistered smaller exact schedule/reservation or independently tested guard amendment to4800seconds. Do not claim full schedule fits3600 or start science against a knowingly insufficient ledger. Numerical/static stages can still proceed independently if admitted within remaining effective cap. All retained outputs must stay below effective24GiB unless amendment verified.
