# Run notes: Qwen3.5-0.8B, 25-pass budget (20 used)

Map built during the run (this model had none): 24 blocks, width 1024; blocks 3, 7, 11, 15, 19, 23 use ordinary attention (8 heads), the rest a linear-attention mixer (16 heads); every MLP is gated (`silu(gate x) * (up x)`, 3,584 neurons) with no bias terms; output matrix tied to the embedding; final norm scales by `1 + weight`. With no biases, a hand-written neuron gets its threshold from a steady direction in the MLP input (mean input, `k.x` close to 1).

| pass | kind | what | outcome |
|---|---|---|---|
| 1 | score | baseline replies on practice (counted as a pass) | 73 of 464 pass |
| - | none | k01 to k03: re-judged saved replies | own tests after the function and wrong function names are the two fixable failures (40 -> 84 of 290 if both fixed) |
| 2 | read | name positions, per-head attention values | name written by heads in blocks 19 and 23; wrong name wins by 1.5 logits on average |
| 3 | read | **edit 1**: scale those heads | fixed 13 name tokens, broke 11. Dropped |
| 4 | score | plain instruction added to the prompt, original weights | 27 -> 37 of 200: the model can follow it |
| 5 | read | same reply under plain and instructed prompts | mean effect of the instruction on the residual stream |
| 6, 7 | read | **edit 2**: always-on neuron adding that effect (block 14) | name tokens wrong 75 -> 44 and 24 |
| 8 | score | edit 2, strength 16 | names fixed (47 -> 21) but answers damaged: 17 -> 10 of 150, maths down. Dropped |
| 9 | read | **edit 3**: same, switched on only at the function name; strength 24 | far too strong (zone effect is 5x larger than the average one) |
| 10 | read | edit 3, strength 4 | name tokens wrong 75 -> 18, 1 broken |
| 11 | score | edit 3, strength 4 | 17 -> 24 of 150, but 14 syntax errors (test arguments copied into the signature) |
| 12, 13 | read | edit 3, strengths 2 and 3 | 2 keeps most of the benefit |
| 14 | score | edit 3, strength 2 | 17 -> 25 of 150, no new syntax errors. Kept |
| 15 | read | last block on full replies | "about to start own tests" is linearly readable (91% caught, 10 false fires in 80,840) |
| 16 | score | edit 3 + **edit 4** (block 23 neuron pushes the closing fence) | 17 -> 33 of 150; maths 31 -> 29 of 50 |
| 17 | read | maths replies | neither neuron fires on maths |
| 18 | score | steeper switches, fresh problems | 23 -> 38 of 140, none lost; maths 38 -> 36 of 60, same replies changed as before |
| 19 | read | all positions incl. prompts | maths change comes from the overwritten neurons not being idle outside code replies; re-picked by activity over prompts, maths and code (k20) |
| 20 | score | final check, held-back practice | 33 -> 55 of 174, none lost; maths 14 -> 14 of 26 |

Frozen edit: 2 neurons, 6,144 weights in 6 tensors.
- Block 14, neuron 418: fires from `def` to the end of the function name; adds 2x the instruction's mean effect at those positions.
- Block 23, neuron 2621: fires when the next token would start the model's own tests after the function; raises the closing code fence over `#` / `assert`.
