# Discovery binding and proposed evaluation design — NOT FROZEN

Coordinator-selected discovery request: model `Gemma-2-2B` BASE, hosted source set `GEMMASCOPE-TRANSCODER-16K`; prompt exactly `The capital of the state containing Dallas is` with no trailing space. Intended output is ` Austin`; actual token IDs, positions, graph and deployed revisions remain pending. Requested/documented defaults are 10 logits, .95 desired probability, 5000 nodes, .8 node threshold and .85 edge threshold; these are not verified effective settings.

This is a Neuronpedia documented example, so the contribution is replication/integration and independent preregistered validation. The methods literature already contains intervention validation; no claim of first validation or an unvalidated field is warranted. Research source: `evidence/feature-attribution-pilot/research/feasibility.md`, parent-supplied SHA256 `afe599fc625c8bfdc07b71b2ebf09d016c899c0e677b2b00f1a5e8707c9bd363`. The research document was read; this author did not repeat its external source review.

## Conditional prompt bank

The following is a text-only proposal suitable if the actual graph selects a feature hypothesized to mediate the Texas → Austin relation. It is not valid for an arbitrary feature (for example a general capital-retrieval feature). Select/finalize it only from graph identity/semantics, before running effects or scoring held-out baselines. Do not call this bank frozen. Dallas is discovery-only and excluded. Token arrays, exact output tokenization and edit positions require independent binding before admission. The candidate directions are Texas-city → California-city and reverse, with the intended fixed Austin/Sacramento contrast; reject ambiguous tokenization rather than silently substituting outputs.

| ID | Texas prompt | California counterpart |
|---|---|---|
| H01 | The capital of the state containing Houston is | The capital of the state containing Los Angeles is |
| H02 | The capital of the state containing San Antonio is | The capital of the state containing San Francisco is |
| H03 | The capital of the state containing El Paso is | The capital of the state containing San Diego is |
| H04 | The capital of the state containing Fort Worth is | The capital of the state containing San Jose is |
| H05 | The capital of the state containing Lubbock is | The capital of the state containing Fresno is |
| H06 | The capital of the state containing Amarillo is | The capital of the state containing Bakersfield is |
| H07 | The capital of the state containing Corpus Christi is | The capital of the state containing Long Beach is |
| H08 | The capital of the state containing Waco is | The capital of the state containing Modesto is |
| H09 | The capital of the state containing Beaumont is | The capital of the state containing Stockton is |
| H10 | The capital of the state containing Tyler is | The capital of the state containing Oakland is |
| H11 | The capital of the state containing Midland is | The capital of the state containing Riverside is |
| H12 | The capital of the state containing Odessa is | The capital of the state containing Anaheim is |
| H13 | The capital of the state containing Abilene is | The capital of the state containing Santa Ana is |
| H14 | The capital of the state containing Wichita Falls is | The capital of the state containing Irvine is |
| H15 | The capital of the state containing College Station is | The capital of the state containing Oxnard is |
| H16 | The capital of the state containing Galveston is | The capital of the state containing Santa Rosa is |

A fixed feature-position rule must follow actual tokenization (for example final city-token position only if supported by the chosen graph), not assume equal prompt lengths. Donor and recipient alignment must be explicit. This fixed bank supports only a narrow prompt-template contrast, not a population/general-language claim.

Candidate non-target prompts, with intended answer contrasts to bind before scoring: `The capital of France is` (Paris/London); `The capital of Japan is` (Tokyo/Paris); `Two plus two equals` (four/five); `The opposite of hot is` (cold/warm). These are narrow controls, not a comprehensive capability suite. Freeze actual token IDs and edit-position policy; an inactive feature on these prompts is only limited specificity evidence. A semantic counterpart feature and residual random-direction control remain necessary where supported.

## Numerical proposal principles and illustrative starting values

The values below are proposals made before seeing graph/effect scores. They cannot be admitted until the actual claim, metric scale, sample count and implementation are fixed. They must not be chosen by graph edge strength or adjusted to measured intervention success.

For a Texas-relation claim, a defensible small fixed-corpus primary score is the signed decrease in `logit(Austin) - logit(Sacramento)` under target ablation, paired per Texas item. Candidate minimum median decrease: 0.5 logit; candidate minimum median target-minus-matched-feature-control effect: 0.25 logit; directional consistency: at least 12 of 16 Texas items. These are practical effect-size criteria, not statistical significance or prior empirical estimates. Feature activity and all-16-item eligibility must be frozen; a low eligible count is INCOMPLETE. Include reverse/donor-patch contrasts with separately fixed sign and minimum before claiming bidirectional mediation.

A rescue candidate criterion is recovery of at least 80% of the ablation-induced contrast loss, only if each denominator exceeds a frozen floor (candidate 0.1 logit); retain absolute recovery and classify unstable denominators INCOMPLETE. Technical restoration should instead match the baseline within the numerical fidelity gate. Non-target maximum candidate absolute contrast change: 0.1 logit per item. The final protocol must decide whether these controls are primary validity/claim requirements and must freeze all rules before testing.

Do not invent original/replacement numeric fidelity ceilings from these effect thresholds. Original numerical tolerances depend on the exact backend/precision and must be fixed before measurements. Replacement reconstruction and logit-error ceilings must be tight enough to distinguish the proposed causal effect from approximation error, with separate residual and output metrics and a documented rationale. Failure to obtain that distinction blocks an original-model claim. Hosted generation access alone supplies neither precision/provenance nor faithful intervention evidence.

Current status: no claim/feature selected, no graph observed by this author, no tokens scored, no thresholds frozen. The graph may invalidate this conditional bank; any replacement must be justified by discovery evidence and recorded before causal/evaluation measurements.

## Actual graph receipt received after drafting

Graph URL: https://www.neuronpedia.org/gemma-2-2b/graph?slug=thecapitalofthes-1795072198897 . Raw artifact: `evidence/feature-attribution-pilot/graph-ui/graph-with-labels.json`; coordinator-reported SHA256 `f36a28b2d4802ab6a555fb918f4ee5fcf78a33aaf5a08940706c638e4c5393ad`. This author inspected its metadata and target-logit record. It reports generator version `405db9a5`, generation batch size 48, requested generation/pruning defaults as above, source set `gemmascope-transcoder-16k`, and prompt pieces `<bos>`, `The`, ` capital`, ` of`, ` the`, ` state`, ` containing`, ` Dallas`, ` is` at indices 0–8. Target node `27_22605_8` is a logit node with token ID 22605, label `Output " Austin" (p=0.330)` and probability 0.330078125. This is graph-provided baseline metadata, not a new score measured by this author.

Exact model weight revision, tokenizer revision/files, transcoder release/files and execution dtype remain unresolved. `metadata.transcoder_list` is empty. Individual feature records use the text `cross layer transcoder`; that field alone cannot establish the actual architecture or intervention semantics of the hosted source set. Graph-provided token pieces do not establish the full input ID array. No feature candidate was selected by this author; graph/operator candidate evidence is still awaited. Earlier “no graph observed” wording describes the chronological state before this receipt, not the present state.

## Corpus size and aggregation amendment before scoring

The proposal now uses 16 distinct city pairs, following the project default in `specs/003-model-dissection/spec.md`; no corpus-size exception is requested. The 16 Texas prompts and their paired California counterparts are 16 evaluation units, not 32 independent samples. They share one template and two states, so they do not supply 16 independent semantic tasks or a representative population. Report per-pair results and finite-corpus medians. If a bidirectional metric is used, first average the two prespecified directional effects within each pair, then take the median over 16 pairs; do not pool 32 directions as independent observations. For the proposed Texas ablation primary, each unit is the Texas member and controls are paired within that unit.

A descriptive paired bootstrap may resample the 16 whole pair records with replacement, PCG64 seed 20261119, 10,000 replicates, reporting percentile 2.5/97.5 intervals for the median effect and paired control advantage. It does not establish generalization across states/templates, override a failed effect-size threshold, or alter the fixed-corpus verdict. All 16 pairs must meet frozen eligibility; fewer yields INCOMPLETE. No prompt duplication or adaptive replacement is allowed.
