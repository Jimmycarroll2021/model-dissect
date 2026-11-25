# Editor notebook: Pythia-14m, 25 passes

Plan: practice code problems 0-299 are for exploring, 300-463 are held back by me to choose the final edit. Non-code practice texts 0-299 guard against side effects.

## Pass 1 (read, original, code:0-299 + noncode:0-299)
Why: see where the model's code predictions go wrong (which kinds of token) and get the per-component values needed to apply the computation map (docs/transfer-002-computation-map.md: per-head packed attention values through W_o, MLP activations through W_down, final LayerNorm, embed_out).
Found (kernel 001): on practice code the model gets 43.2% of tokens. Biggest systematic losses: it never predicts `def` at the start of the code (1.6% of all code tokens; it predicts `#` or a newline), `):` closing a parameter list (1.75%; it predicts `,`), the ` \n` line ending MBPP uses (1.5%), `return` (1.5%), `_` inside names (2%). It over-predicts `self`, ` __` and `#`, which suggests a prior for class methods and comments that is wrong for these top-level functions.

## Kernel 002: how close are the misses?
Why: an edit can only flip a prediction if the right token is already near the top. Applying the final LayerNorm and embed_out (map: "final γ_f, β_f ... output projection") to the saved final residual gives the full scores without running the model.
Found (kernel 002): 57% of code tokens are wrong; for 21% of all code tokens the right token is within 2 logits of the top, so modest, targeted changes could flip many. `def` at the start of the code is a far miss (median 4.7 logits behind `#`); `):`, `_`, `return`, `)`, `if` are near misses.

## Kernel 003: which components push the wrong way on code?
Why: the computation map gives each head's write as its 32 packed values through its 32 columns of W_o, and each MLP's write as activations through W_down; with the final LayerNorm frozen at its measured scale these can be read out through embed_out as a contribution to (right token minus best wrong token). A component that is negative on code but not on English is a candidate to turn down.
Found (kernel 003): by direct effect the last MLP (layer 5) is the one component that helps English (+0.31 logits of margin per token) and hurts code (-0.29, and -0.62 on the tokens the model gets wrong). Nothing sits after it except the final norm and the output layer, so its direct effect is its whole effect. Layer 3's MLP is negative on both, but it feeds later layers so its direct effect is not the whole story.

## Kernel 004: which layer-5 MLP neurons do the damage, and what do they promote?
Why: the map says the MLP writes `GELU(B) W_down^T`, one column of W_down per neuron, so the layer's harm on code can be split neuron by neuron and each neuron's column read through embed_out to see which tokens it pushes.
Found (kernel 004): the harm is spread over many layer-5 neurons; 30 of the 512 each cost code more than 0.01 logits of margin per token, together 0.52, and the same 30 are slightly negative on English too (-0.10 in total). Their output columns do not read as clean token lists through embed_out, so I cannot name what each one "means"; the evidence is the measured direct effect only.

## Kernel 005: is that neuron list stable, or noise?
Why: 30 picked out of 512 could be chance. Check that the same neurons are negative on two disjoint halves of the practice problems before touching them.
Found (kernel 005): the per-neuron effect on code correlates 0.94 between two halves of the problems, so it is not noise. 26 neurons are negative in both halves, below -0.01 overall and not helpful to English.

## Edit 1 + Pass 2 (score, both, code:300-463 + noncode:0-299)
Edit: zero the 26 columns of `gpt_neox.layers.5.mlp.dense_4h_to_h.weight` for those neurons (switch off what they write). Expected from the direct effect: about +0.45 logits of margin per code token and a small plus on English. Test on the 164 practice problems I have not looked at, and on English.
Result (pass 2): WRONG. Code accuracy on the held-back problems fell 3.1 points (41.9% -> 38.8%, se 0.4); English fell 0.3. Edit 1 is reverted.
Why it failed, my reading: I ranked neurons by their average effect on the margin over all tokens. The average is dominated by tokens the model misses by a wide margin, where no small change can flip the answer. What matters for accuracy is the effect on tokens near the decision. These neurons evidently help there.

## Kernel 007: the same attribution, restricted to tokens near the decision
Why: redo the component and neuron tables counting only tokens where the right token is within 1.5 logits of the best alternative (either side). A component that is negative there is costing flips.
Found (kernel 007): near the decision the last MLP is positive on code (+0.15), and the 26 neurons I switched off were only mildly negative there (-0.08), yet accuracy fell 3 points. So even the restricted attribution does not predict the outcome. Reason: it measures the right token against one rival; switching a neuron off moves every token's score, and these neurons mostly hold down other wrong tokens. Lesson: pairwise direct attribution is not a safe basis for pruning. Stop pruning by attribution.

New approach: fix named, systematic mistakes instead. The clearest is the first code token: after the `# description` line the code starts with `def` (232 of 300), `import` or `from`, and the model predicts `#` 252 times and a newline 48 times, never the right token; that alone is 1.6% of all code tokens. Plan: write one MLP neuron by hand in layer 5 that detects "end of the opening comment line" and writes the `def` direction.

## Kernel 008: can that position be detected from the layer-5 MLP input?
Why: the map says the layer-5 MLP reads N2 = LayerNorm(X; post_attention_layernorm) of the block input and neuron n fires on W_up[n,:]. Take the difference of mean N2 between the code-start positions and all other positions, and see how cleanly that direction separates them.
Found (kernel 008): along the mean-difference direction (taken from half the problems) the code-start positions sit at 17.5 to 23.7 (other half: all above 20.9) and every other position, code or English, is below 16.4. Clean. Least-used layer-5 neuron: 108.

## Edit 2 + Pass 3 (score, both, code:300-463 + noncode:0-299)
Edit: rewrite layer-5 neuron 108 by hand. Input row = 1.5 x that direction, bias so it switches on above 19.5 (silent elsewhere). Output column = the direction that raises `def` against the tokens the model currently predicts there, sized to lift `def` about 8 logits over `#` when the neuron is on. Expect the first code token to become right on problems starting with `def` (about 77% of them), nothing else to move.
Result (pass 3): +1.26 points on held-back code (41.9% -> 43.2%, se 0.12), English unchanged (+0.004). Edit 2 kept. The hand-written-neuron method works: detect a context by mean difference, write the right token's direction.
Caveat for the write-up: this fixes the model's handling of the `# description` then `def` layout used here; it is format knowledge, not reasoning about code.

## Kernel 010: which other fixed contexts have a predictable token the model misses?
Why: look for contexts where one token is right most of the time but the model predicts something else. Contexts tried: position counted from the `def` token on the function-definition line, and what follows specific tokens.
Found (kernel 010): one more fixed context stands out: the token right after the first piece of a function name is `_` 86% of the time (snake_case names) and the model says `(` 66% of the time; a rule there is worth about +0.9% of code tokens. Other contexts are worth 0.3% or less each. Separately the ` \n` line ending (space before newline) costs 1.5% and the model predicts plain `\n`.

## Kernel 011: are those two contexts detectable at the layer-5 MLP input?
Why: same test as kernel 008, direction from one half of the problems, separation checked on the other half.
Found (kernel 011): at the layer-5 MLP input neither context separates with one direction: catching 80% of "one token after def" fires on about 200 other positions, and the ` \n` ending is worse; line-ending style is also mixed inside 93 of 300 problems, so that one is dropped.

## Kernel 012: is "one token after def" cleaner at an earlier layer's MLP input?
Why: "the previous token was def" is local information that early layers should hold more plainly than layer 5. Every block's MLP reads the same kind of input (map: N2 = LN(X; post_attention_layernorm) per block) and writes to the residual stream that reaches the output.
Found (kernel 012): at the layer-3 MLP input "one token after def" separates almost perfectly (catching 95% fires on 2 other positions out of ~21,000); layers 0-2 and 4-5 are far worse. So the rule belongs in layer 3.

## Kernel 013: other two-token contexts with a dominant next token the model misses
Why: before spending passes, list every context of the form (token before last, last token) where one next token dominates and the model does worse than that; counted on half the problems, checked on the other half.
Found (kernel 013): apart from the code-start rule already installed (+1.2% of tokens), two-token contexts are each worth little: `return` after a shallow indent +0.6% in total, the rest 0.2% or less. With the `_` rule (about +0.9%) the reachable total on this layout is roughly 2.5 to 2.8 points, so the 3-point mark set in advance may not be reached. Continue anyway and report what is measured.

## Edit 3 + Pass 4 (score, both, code:300-463 + noncode:0-299)
Edit: hand-write a second neuron, this one in layer 3 (least-used neuron there): fires one token after `def` (mean-difference detector at the layer-3 MLP input), writes the direction that raises `_` over `(`. Unknown: layers 4 and 5 also read what it writes, so the effect at the output may differ from the direct path. That is what the pass is for.
Result (pass 4): +1.76 points on held-back code (se 0.15), English +0.08. The second neuron added +0.5, about half of what the rule could give. Likely the 3-logit lift is too small once layers 4 and 5 have had their say.

## Edit 3b + Pass 5 (same sets)
Edit: same layer-3 neuron, lift doubled to 6 logits. One adjustment, not a sweep.
Result (pass 5): +2.06 points on held-back code (se 0.17), English +0.08. Keep the 6-logit version.

## Kernel 016: three more generic Python contexts, detectability by layer
Why: candidates that are ordinary Python rather than this dataset's naming habits: (i) `for i in` -> ` range`; (ii) the line after a top-of-file import -> `def`; (iii) first-level indented line start -> `return`. For each: how often the rule token is right, how often the model is right now, and at which layer's MLP input the context separates.
Found (kernel 016): (iii) is the valuable one: at a first-level indented line start `return` is right 30% of the time, the model is right 12% and says `"""` (a docstring) 21% of the time, which these functions never have; a rule is worth about +0.7%. It separates best at layer 2 (23 false fires for 90% caught). (ii) the line after an import is `def` 91% of the time, model 0%, +0.3%, separates with no false fires at layer 4. (i) `for i in` -> ` range` +0.16%, layer 3.

## Edit 4 + Pass 6 (same sets)
Edit: three more hand-written neurons: layer 2 (indent start: raise `return` over `"""`, moderate 3-logit lift so it only tips close calls), layer 4 (after an import line: `def`), layer 3 (`for i in`: ` range`).
Result (pass 6): +1.98 on held-back code (se 0.19), English -0.16. Worse than before the three additions (+2.06, English +0.08). At least one of the three hurts.
Suspect: the `return` rule. Its context is ambiguous (`return` right only 30%) and its detector has near-threshold negatives; GELU also gives small negative outputs just below threshold, which push the wrong way on many tokens.

## Edit 4b + Pass 7 (same sets)
Edit: restore layer 2 to the original (drop the `return` neuron), keep the other two.
Result (pass 7): +2.09 on held-back code (se 0.18), English -0.14. So the `return` neuron was costing 0.11 on code: dropped for good. But the import and range neurons together add only +0.03 on code and English went from +0.08 to -0.14. Something is off with one of them.

## Pass 8 (read, edited, code:0-149 + noncode:0-149)
Why: compare token by token with the original predictions saved in pass 1 on the same texts, to see exactly which tokens each neuron changes, for better or worse, and where on English it leaks.
Found (pass 8, token-by-token): code-start neuron fixes 92 of 149 starts and breaks none; underscore neuron fixes 77, breaks 5; range neuron fixes 5, breaks 1. The import neuron (layer 4) fires at the right 23 places but changes nothing: its write is damped by layer 5, so it needs to live in layer 5 where the write is direct. On English no neuron fires, yet 423 predictions change (44 fixed, 60 broken): that is the cost of overwriting four lightly used neurons, about -0.2 points, inside the allowance.

## Kernel 020: detectability of the remaining small contexts, by layer
Why: move the import rule to layer 5 and see which of the small two-token contexts from kernel 013 can be detected cleanly enough to be worth a neuron.
Found (kernel 020; the layer-0 column is meaningless, it only sees the current token): the import rule separates at layer 5 with 1 false fire. Small contexts that separate cleanly: `(test` -> `_` (layer 3), `res)` -> space, `(res` -> `)`, top-of-file `import` -> ` re` (layer 5). These last ones are habits of this problem set's authors (a variable called `res`, arguments called `test_...`, `import re`), not general Python; they are worth about +0.6% together.

## Edit 5 + Pass 9 (same sets)
Edit: drop the layer-4 import neuron; add layer-5 neurons for: after an import line -> `def`; `res)` -> space; `(res` -> `)`; top-of-file `import` -> ` re`; and a layer-3 neuron for `(test` -> `_`. Raise the code-start lift from 8 to 11 (it still lost to `"""` on some starts).
Result (pass 9): +2.68 on held-back code (se 0.22), English +0.15. Kept.

## Edit 6 + Pass 10 (same sets), last round
Edit: three more small rules: `return (` -> `res` and `_t` -> `up` (tuple) in layer 3, ` = 0` -> newline in layer 5. If this does not beat +2.68 it is reverted.
Result (pass 10): +2.54 on held-back code, English 0.00. Worse than pass 9. Reverted to the pass-9 edit.

## Final choice (10 of 25 passes used)
Stop here. The remaining contexts are each worth under 0.1% and the last round showed more rules start to cost more than they give. Final edit = the pass-9 state: 8 hand-written MLP neurons (layer 5: code start -> def; after an import line -> def; `res)` -> space; `(res` -> `)`; `import` -> ` re`; layer 3: one token after def -> `_`; `for i in` -> ` range`; `(test` -> `_`).
On the held-back practice problems: +2.68 points of code-token accuracy, English +0.15. This is below the 3-point mark set in advance, so I expect the held-out verdict to be "no gain by the preset rule" with a real, smaller improvement.
What the edit is, plainly: layout and naming habits of this problem style written into spare neurons. It does not make the model reason better about code.
