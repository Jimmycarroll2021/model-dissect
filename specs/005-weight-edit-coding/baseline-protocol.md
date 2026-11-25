# Baseline protocol (frozen before any baseline run)

Frozen 2026-11-25 together with `exploratory/weightedit/{common.py,fetch_data.py,baseline.py}`.

**Candidates** (weights as held in the earlier workspaces; hashes recorded in the result file):
MiniMind `full_sft_768.pth`, Pythia-14m `model.safetensors`, GPT-2 `model.safetensors`. fp32, one GPU, greedy decoding only.

**Problems.** MBPP (`google-research/mbpp/mbpp.jsonl`, 974 problems) and HumanEval (164).
- Practice: MBPP task_id 511 to 974 (validation and train, 464 problems). The only problems used for the baseline and, later, by the editor.
- Held-out: MBPP task_id 11 to 510 (500 problems) and all of HumanEval. Not scored until an edit is frozen.
- MBPP task_id 1 to 10 are unused. Code is normalised to `\n` line ends.

**Level A, whole problems.** Prompt: `"""\n{text}\n{first test}\n"""\n`. Greedy, at most 256 new tokens, stopped at the first of `\nclass `, `\nassert`, `\n"""`, `\nprint(`, `\nif __name__`, `\n#`. The completion followed by the problem's test asserts runs in a sandbox (no network, 10 s, 1 GB). Pass = exit code 0. A failed answer gets the first type that applies: no code produced (empty or no `def`), repetition loop (a non-empty line repeated 4 or more times), cut at token limit, syntax error, timeout, runtime error, wrong answer (assertion failed).

**Level B, code completion.** Text: `# {text}\n{code}`, tokenised by the model's own tokenizer with its start token if it has one, truncated to the model's context. Teacher-forced. Scored tokens: those that start at or after the first character of the code. Per problem: top-1 accuracy over scored tokens. Reported: mean over problems (primary), and bits per code character (descriptive; comparable across tokenizers).

**Non-code check.** The first 300 GSM8K train questions as plain text, same top-1 accuracy over all tokens after the first.

**Selection rule.** Take the hardest level first. A candidate qualifies at Level A if its practice pass rate is between 5% and 60%; at Level B if its mean practice top-1 accuracy is between 5% and 60%. The level used is A if any candidate qualifies there, otherwise B. The chosen model is the qualifying candidate with the fewest parameters. If none qualifies at either level: stop and report.

**Repeat.** Run twice; result files must be identical.
