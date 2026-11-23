# Paraphrase v1: per-edge results (5 paraphrases per graph, final-position read and removal)

Edges tested: 73; position not transferable: 2. Agreement of own-prompt category with paraphrase majority: 67/73 = 0.92. Per-paraphrase outcomes over valid pairs: DISTINCT 136, NOT_DISTINCT 132, WRONG_SIGN 51, INACTIVE 5.

| graph | edge | own verdict | own E | valid | active | distinct | median E | median drop in p | D / ND / WS / IN | majority | agrees |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-addition | 24/14176 | NOT_DISTINCT | 1.37 | 5 | 1.00 | 0.20 | 1.42 | 0.01 | 1 / 3 / 1 / 0 | not distinct | yes |
| gemma-addition | 25/14682 | HOLDS | 3.86 | 5 | 1.00 | 1.00 | 3.38 | 0.21 | 5 / 0 / 0 / 0 | distinct | yes |
| gemma-addition | 24/2998 | WEAKER | 1.11 | 5 | 1.00 | 0.40 | 0.29 | 0.06 | 2 / 1 / 2 / 0 | not distinct | NO |
| gemma-fact-dallas-austin | 20/15589 | WEAKER | 1.37 | 4 | 1.00 | 1.00 | 1.68 | 0.09 | 4 / 0 / 0 / 0 | distinct | yes |
| gemma-fact-dallas-austin | 21/5943 | HOLDS | 3.30 | 4 | 1.00 | 1.00 | 2.80 | 0.15 | 4 / 0 / 0 / 0 | distinct | yes |
| gemma-fact-dallas-austin | 23/12237 | HOLDS | 1.81 | 4 | 1.00 | 1.00 | 1.86 | 0.08 | 4 / 0 / 0 / 0 | distinct | yes |
| gemma-fact-oakland-sacramento | 21/5943 | HOLDS | 2.64 | 4 | 1.00 | 1.00 | 2.34 | 0.07 | 4 / 0 / 0 / 0 | distinct | yes |
| gemma-fact-oakland-sacramento | 19/9209 | WEAKER | 1.16 | 4 | 1.00 | 1.00 | 1.76 | 0.08 | 4 / 0 / 0 / 0 | distinct | yes |
| gemma-fact-oakland-sacramento | 24/13277 | NOT_DISTINCT | 1.65 | 4 | 1.00 | 0.00 | 1.81 | 0.04 | 0 / 4 / 0 / 0 | not distinct | yes |
| gemma-small-big-en | 23/8683 | WEAKER | 1.09 | 5 | 1.00 | 1.00 | 1.17 | 0.02 | 5 / 0 / 0 / 0 | distinct | yes |
| gemma-small-big-en | 23/14350 | NOT_DISTINCT | 0.85 | 5 | 1.00 | 0.00 | 0.80 | 0.01 | 0 / 4 / 1 / 0 | not distinct | yes |
| gemma-small-big-en | 21/10062 | NOT_DISTINCT | 0.02 | 5 | 0.80 | 0.00 | 0.01 | 0.02 | 0 / 3 / 1 / 1 | not distinct | yes |
| gemma-small-big-fr | 23/14350 | NOT_DISTINCT | 0.73 | 4 | 1.00 | 0.00 | 0.70 | 0.09 | 0 / 4 / 0 / 0 | not distinct | yes |
| gemma-small-big-fr | 24/4707 | NOT_DISTINCT | 0.13 | 4 | 1.00 | 0.00 | 0.04 | 0.02 | 0 / 4 / 0 / 0 | not distinct | yes |
| gemma-small-big-fr | 24/16045 | HOLDS | 1.35 | 4 | 1.00 | 1.00 | 1.36 | 0.07 | 4 / 0 / 0 / 0 | distinct | yes |
| gemma-G | 25/5604 | HOLDS | 2.71 | 5 | 1.00 | 1.00 | 2.54 | 0.22 | 5 / 0 / 0 / 0 | distinct | yes |
| gemma-G | 22/12304 | NOT_DISTINCT | 0.51 | 5 | 1.00 | 0.40 | 0.61 | 0.03 | 2 / 3 / 0 / 0 | not distinct | yes |
| gemma-G | 23/14585 | WRONG_SIGN | -0.04 | 5 | 1.00 | 0.00 | 0.01 | 0.01 | 0 / 3 / 2 / 0 | not distinct | yes |
| gemma-Mexico-Spanish | 22/12284 | WEAKER | 1.76 | 5 | 1.00 | 1.00 | 1.98 | 0.12 | 5 / 0 / 0 / 0 | distinct | yes |
| gemma-Mexico-Spanish | 19/13366 | NOT_DISTINCT | 1.13 | 5 | 1.00 | 0.20 | 1.69 | 0.16 | 1 / 4 / 0 / 0 | not distinct | yes |
| gemma-Mexico-Spanish | 24/410 | NOT_DISTINCT | 0.79 | 5 | 1.00 | 0.40 | 0.80 | 0.07 | 2 / 3 / 0 / 0 | not distinct | yes |
| gemma-basket | 21/4818 | NOT_DISTINCT | 1.57 | 5 | 1.00 | 0.00 | 1.53 | 0.24 | 0 / 5 / 0 / 0 | not distinct | yes |
| gemma-basket | 21/9324 | NOT_DISTINCT | 1.49 | 5 | 1.00 | 0.40 | 1.40 | 0.06 | 2 / 3 / 0 / 0 | not distinct | yes |
| gemma-basket | 23/3604 | HOLDS | 1.05 | 5 | 1.00 | 1.00 | 0.98 | 0.13 | 5 / 0 / 0 / 0 | distinct | yes |
| gemma-bee-insect | 22/16038 | WRONG_SIGN | -0.45 | 5 | 1.00 | 0.00 | -0.54 | 0.03 | 0 / 0 / 5 / 0 | not distinct | yes |
| gemma-bee-insect | 14/11504 (earlier token) | WRONG_SIGN | -0.44 | | | | | | | position not transferable | |
| gemma-bee-insect | 19/5460 | WRONG_SIGN | -1.00 | 5 | 1.00 | 0.00 | -0.80 | -0.01 | 0 / 0 / 5 / 0 | not distinct | yes |
| gemma-cat-hat | 25/10571 | HOLDS | 1.32 | 5 | 1.00 | 1.00 | 1.32 | 0.04 | 5 / 0 / 0 / 0 | distinct | yes |
| gemma-cat-hat | 24/1512 | HOLDS | 1.04 | 5 | 1.00 | 1.00 | 1.03 | 0.04 | 5 / 0 / 0 / 0 | distinct | yes |
| gemma-cat-hat | 22/10393 | HOLDS | 0.55 | 5 | 0.80 | 0.40 | 0.37 | 0.03 | 2 / 2 / 0 / 1 | not distinct | NO |
| gemma-dollar | 18/10102 | WRONG_SIGN | -0.52 | 5 | 1.00 | 0.00 | -0.36 | 0.00 | 0 / 0 / 5 / 0 | not distinct | yes |
| gemma-dollar | 20/5796 | WRONG_SIGN | -0.92 | 5 | 1.00 | 0.00 | -1.03 | 0.00 | 0 / 0 / 5 / 0 | not distinct | yes |
| gemma-dollar | 18/10914 | NOT_DISTINCT | 0.34 | 5 | 1.00 | 0.00 | -0.09 | 0.02 | 0 / 2 / 3 / 0 | not distinct | yes |
| gemma-english | 25/6667 | NOT_DISTINCT | 0.97 | 4 | 1.00 | 0.00 | 0.76 | 0.15 | 0 / 4 / 0 / 0 | not distinct | yes |
| gemma-english | 23/2040 | WEAKER | 0.40 | 4 | 1.00 | 0.00 | 0.11 | 0.00 | 0 / 4 / 0 / 0 | not distinct | NO |
| gemma-english | 19/13366 | WRONG_SIGN | -0.13 | 4 | 1.00 | 0.00 | -0.45 | -0.01 | 0 / 0 / 4 / 0 | not distinct | yes |
| gemma-euro | 23/7512 | NOT_DISTINCT | 0.56 | 3 | 1.00 | 0.00 | 0.53 | 0.00 | 0 / 3 / 0 / 0 | not distinct | yes |
| gemma-euro | 21/467 | NOT_DISTINCT | 0.89 | 3 | 1.00 | 0.33 | 1.08 | 0.03 | 1 / 2 / 0 / 0 | not distinct | yes |
| gemma-euro | 23/6406 | WEAKER | 0.67 | 3 | 1.00 | 1.00 | 0.73 | 0.02 | 3 / 0 / 0 / 0 | distinct | yes |
| gemma-fact-shanghai-beijing | 21/5943 | HOLDS | 2.67 | 4 | 1.00 | 1.00 | 2.44 | 0.11 | 4 / 0 / 0 / 0 | distinct | yes |
| gemma-fact-shanghai-beijing | 19/12274 | WEAKER | 1.10 | 4 | 1.00 | 1.00 | 1.25 | 0.08 | 4 / 0 / 0 / 0 | distinct | yes |
| gemma-fact-shanghai-beijing | 19/12274 (earlier token) | WEAKER | 0.67 | | | | | | | position not transferable | |
| gemma-fact-vancouver-victoria | 21/2236 | NOT_DISTINCT | 2.16 | 5 | 1.00 | 0.00 | 2.35 | 0.10 | 0 / 5 / 0 / 0 | not distinct | yes |
| gemma-fact-vancouver-victoria | 19/15123 | NOT_DISTINCT | 0.13 | 5 | 1.00 | 0.00 | 0.39 | 0.04 | 0 / 5 / 0 / 0 | not distinct | yes |
| gemma-fact-vancouver-victoria | 24/13277 | NOT_DISTINCT | 0.83 | 5 | 1.00 | 0.00 | 1.16 | 0.02 | 0 / 5 / 0 / 0 | not distinct | yes |
| gemma-girl-is | 25/11727 | HOLDS | 1.11 | 4 | 0.75 | 0.75 | 1.09 | 0.02 | 3 / 0 / 0 / 1 | distinct | yes |
| gemma-girl-is | 24/13277 | NOT_DISTINCT | 0.51 | 4 | 1.00 | 0.00 | 0.62 | 0.00 | 0 / 4 / 0 / 0 | not distinct | yes |
| gemma-girl-is | 24/13814 | HOLDS | 0.72 | 4 | 0.75 | 0.75 | 0.77 | 0.01 | 3 / 0 / 0 / 1 | distinct | yes |
| gemma-girls-are | 22/11517 | HOLDS | 3.51 | 4 | 1.00 | 1.00 | 2.61 | 0.03 | 4 / 0 / 0 / 0 | distinct | yes |
| gemma-girls-are | 25/9974 | HOLDS | 2.02 | 4 | 1.00 | 1.00 | 0.98 | 0.01 | 4 / 0 / 0 / 0 | distinct | yes |
| gemma-girls-are | 24/13277 | NOT_DISTINCT | 0.66 | 4 | 1.00 | 0.50 | 0.75 | 0.00 | 2 / 2 / 0 / 0 | not distinct (tie) | yes |
| gemma-gp-nps | 22/7306 | HOLDS | 1.59 | 3 | 0.67 | 0.67 | 1.73 | 0.02 | 2 / 0 / 0 / 1 | distinct | yes |
| gemma-gp-nps | 21/11913 | WEAKER | 0.88 | 3 | 1.00 | 1.00 | 0.94 | 0.01 | 3 / 0 / 0 / 0 | distinct | yes |
| gemma-gp-nps | 18/4181 | HOLDS | 3.89 | 3 | 1.00 | 1.00 | 3.87 | 0.05 | 3 / 0 / 0 / 0 | distinct | yes |
| gemma-keys-cabinet | 22/11517 | WEAKER | 1.70 | 4 | 1.00 | 1.00 | 1.93 | 0.03 | 4 / 0 / 0 / 0 | distinct | yes |
| gemma-keys-cabinet | 23/4388 | HOLDS | 3.15 | 4 | 1.00 | 1.00 | 4.15 | 0.02 | 4 / 0 / 0 / 0 | distinct | yes |
| gemma-keys-cabinet | 0/9026 | STRONGER | 2.86 | 4 | 1.00 | 0.50 | 0.95 | 0.00 | 2 / 0 / 2 / 0 | not distinct (tie) | NO |
| gemma-michael-jordan-es | 21/9324 | NOT_DISTINCT | 0.52 | 5 | 1.00 | 0.00 | 0.51 | -0.04 | 0 / 5 / 0 / 0 | not distinct | yes |
| gemma-michael-jordan-es | 21/4818 | HOLDS | 0.99 | 5 | 1.00 | 0.40 | 1.03 | 0.17 | 2 / 3 / 0 / 0 | not distinct | NO |
| gemma-michael-jordan-es | 25/13416 | NOT_DISTINCT | 0.42 | 5 | 1.00 | 0.00 | 0.39 | 0.11 | 0 / 5 / 0 / 0 | not distinct | yes |
| gemma-michael-jordan | 21/9324 | NOT_DISTINCT | 0.05 | 5 | 1.00 | 0.00 | 0.19 | 0.00 | 0 / 4 / 1 / 0 | not distinct | yes |
| gemma-michael-jordan | 20/9090 | NOT_DISTINCT | 0.16 | 5 | 1.00 | 0.00 | 0.11 | 0.02 | 0 / 3 / 2 / 0 | not distinct | yes |
| gemma-michael-jordan | 21/4818 | NOT_DISTINCT | 0.28 | 5 | 1.00 | 0.00 | 0.29 | 0.02 | 0 / 5 / 0 / 0 | not distinct | yes |
| gemma-saison | 20/9338 | WEAKER | 1.21 | 5 | 1.00 | 0.80 | 1.40 | 0.13 | 4 / 1 / 0 / 0 | distinct | yes |
| gemma-saison | 24/3772 | NOT_DISTINCT | 0.03 | 5 | 1.00 | 0.00 | 0.10 | 0.05 | 0 / 3 / 2 / 0 | not distinct | yes |
| gemma-saison | 24/9342 | NOT_DISTINCT | 1.17 | 5 | 1.00 | 0.00 | 1.18 | 0.03 | 0 / 5 / 0 / 0 | not distinct | yes |
| gemma-small-big-zh | 24/2394 | NOT_DISTINCT | 2.50 | 5 | 1.00 | 0.00 | 2.48 | 0.28 | 0 / 5 / 0 / 0 | not distinct | yes |
| gemma-small-big-zh | 23/8488 | WRONG_SIGN | -0.60 | 5 | 1.00 | 0.00 | -0.51 | 0.02 | 0 / 0 / 5 / 0 | not distinct | yes |
| gemma-small-big-zh | 23/8683 | WEAKER | 0.92 | 5 | 1.00 | 1.00 | 0.94 | 0.01 | 5 / 0 / 0 / 0 | distinct | yes |
| gemma-tom-brady | 23/8855 | WEAKER | 1.11 | 3 | 1.00 | 1.00 | 1.02 | 0.04 | 3 / 0 / 0 / 0 | distinct | yes |
| gemma-tom-brady | 24/13277 | NOT_DISTINCT | 1.33 | 3 | 1.00 | 0.00 | 1.24 | 0.06 | 0 / 3 / 0 / 0 | not distinct | yes |
| gemma-tom-brady | 19/15763 | NOT_DISTINCT | 0.09 | 3 | 1.00 | 0.00 | 0.18 | 0.00 | 0 / 3 / 0 / 0 | not distinct | yes |
| gemma-verano | 20/9338 | WEAKER | 1.17 | 5 | 1.00 | 0.40 | 0.94 | 0.06 | 2 / 3 / 0 / 0 | not distinct | NO |
| gemma-verano | 24/13490 | WRONG_SIGN | -0.13 | 5 | 1.00 | 0.00 | -0.08 | 0.02 | 0 / 0 / 5 / 0 | not distinct | yes |
| gemma-verano | 25/2591 | HOLDS | 1.32 | 5 | 1.00 | 0.60 | 1.38 | 0.10 | 3 / 2 / 0 / 0 | distinct | yes |

D = DISTINCT, ND = NOT_DISTINCT, WS = WRONG_SIGN, IN = INACTIVE (counts over valid paraphrases). Median E and median drop in p count inactive paraphrases as 0. Own E is reference only.
