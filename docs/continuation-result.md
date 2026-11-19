# What longer responses revealed

The four-case response-window diagnostic is complete and independently repeated. Extending observation from24 to256 generated tokens allowed one comparison response to finish; it was graded **WRONG** under the frozen semantic rubric. The other three responses were still **INCOMPLETE**. This resolves part of the earlier censoring question, but does not qualify a task family or explain an internal mechanism. The broader reverse-engineering goal remains active.

The independent review is [PASS_CONTINUATION_INDEPENDENT_REPEAT_AND_GRADING](../evidence/continuation/review/continuation-repeat-final-v1.json), SHA256 `b051bdf6013ff6e7f43e7c5d03b0523e4b5a6ea0990a4c4b21f0ed04841589eb`. This report's exact-document review is separate and pending. The [frozen protocol](../evidence/continuation/response-window-protocol-v1.md) and [candidate manifest](../evidence/continuation/response-window-candidate-v1.json) identify the tested computation and evaluation rules.

## Observed responses

We retained greedy CPU FP32, cache-off, full-prefill computation and the existing input template. Three inputs were the already exposed first COPY, SUBTRACT and COMPARE screening cases; the fourth was the source-selected upstream general-chat example, “你有什么特长？” (“What are your strengths?”). They are diagnostic cases, not a fresh qualification set or a representative population sample.

| Input | Required answer | Observed termination | Independently frozen grade |
|---|---|---|---|
| COPY: ask which digit甲 wrote, after giving0,9,6 | 0 |256tokens, length cutoff | INCOMPLETE |
| SUBTRACT:62 minus62 | 0 |256tokens, length cutoff | INCOMPLETE |
| COMPARE: is50 greater or smaller than2? | 大 (“greater”) | EOS at token256 | WRONG |
| General chat: “What are your strengths?” | Open response |256tokens, length cutoff | INCOMPLETE |

The independent grader committed its assessment before reading the producer's assessment; both agreed on all four grades. COMPARE contains unretracted false mathematical assertions and was classified WRONG under frozen rule3. The other responses remain censored: their visible incorrect assertions or repetition do not turn them into completed-answer failures. No extrapolation is made about what they would eventually say beyond256tokens.

Concrete excerpts show what was observed:

- COPY repeatedly emits “甲只写下数字0，乙只写下数字6” (“甲 only wrote0;乙 only wrote6”), although the input assigned乙 the digit9.
- SUBTRACT emits “62×2=122” and “62-122=12”, then repeats the latter arithmetic passage. These are visible false assertions inside an incomplete response.
- COMPARE repeatedly argues that the answer depends on how50 and2 are defined, ending without the requested valid comparison explanation.
- General chat begins a numbered description of purported assistant capabilities. Items7 and8 repeat “多语言支持” (“multilingual support”); its tenth item is cut off. These self-descriptions are generated text, not independently established capabilities.

The repeated token sequences include exact nonoverlapping spans of54/54/60/23tokens respectively, as recorded by the independent review. Repetition is an observable phenomenon for future investigation, not evidence identifying a circuit. Full decoded text and steps are retained in the producer raw result (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-continuation-001-20261118/runs/continuation-producer-v1.json`; external or omitted historical artifact) and independent raw result (`EXTERNAL_ARTIFACT_ROOT/workspaces/minimind-continuation-001-20261118/runs/continuation-independent-v1.json`; external or omitted historical artifact).

## Fidelity and independent repeat

Before generation, each run checked four saved computational boundaries at synthetic prefix lengths128,256 and319. All12 comparisons per run had maximum absolute error0.0. All24 captured boundary tensor hashes matched across producer and independent execution. This is long-prefix same-mode evidence for these cases, not a repair of the historical cache/full R2 failure or a universal fidelity guarantee.

Across the two runs, all1024 generated token IDs, step records, selected-logit/full-vocabulary hashes, EOS ranks and probabilities matched exactly; generation journals were byte-identical. The three reused prompts reproduced their earlier24-token prefixes and logit hashes exactly. Prepared inputs, admitted artifacts and all three historical ledger chains were unchanged. Thus extending the observation window retained the original beginning of each response.

EOS probabilities at the last observed token were approximately0.00000167145 for COPY,0.000312453 for SUBTRACT,0.269576 for COMPARE and0.00000000225628 for general chat; corresponding ranks147,162,1,2043. The verifier checked bounds, selection and exact repetition. Saved vector hashes alone do not permit independent recomputation of the complete softmax distribution; no stronger verification is claimed.

## Costs, limits and continuation

Each run processed165246 positions in1030 forwards:1024 generation forwards and6 parity forwards. Together they processed330492 positions in2060 forwards. Producer wall time was102.43271083710715seconds and independent wall time103.98392948799301seconds; total206.41664032510016seconds. The phase reserved353280 positions and1800 model seconds. Adding historical450928 reservations gives **804208 lifetime reserved positions**, leaving **195792** under the unchanged1000000 ceiling. Reserved positions differ from actual processed positions, and unused reservation capacity is not refunded.

Phase start1794990938.83567 and deadline1794994538.83567 are fixed. The independent receipt records zero active rows at completion; it is a snapshot, not a claim that later authorized work is idle. Historical clocks and charges remain unchanged. No further run is authorized by this report alone.

The separate computation-map review has passed static checking of all91 tensor keys/shapes and equations ([receipt](../evidence/computation-map/review/receipt.json)); that supplies a source-assisted structural explanation. Dynamic traces and a validated causal explanation are still required by the broader goal. A separately prepared echo investigation may use the reported observation that all16 earlier COPY cases share initial tokens[5078,1226,504,1372] and then emit the input digit at the fifth token. This is a proposed measurable behavior, not an accepted mechanism or a new runtime result. Its exact protocol, controls, fresh evaluation and independent review remain separate.

The original cache/full FAIL, earlier coordinate INCONCLUSIVE and static-OV REFUTED outcomes are preserved. This diagnostic narrows what the model demonstrably generated; it does not establish general inability, broad mathematical competence or full understanding of its weights.
