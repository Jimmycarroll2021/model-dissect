# Baseline protocol (frozen before any baseline run)

Frozen 2026-11-26 together with `exploratory/weightedit/qwen/{qcommon.py,qfetch.py,qbaseline.py,run_baseline.sh}`. Before the freeze the model was loaded once and asked one made-up question (sum of a list) to see its answer format; an 8-problem smoke test checks the plumbing only and its numbers are not used.

**Model.** `Qwen/Qwen3.5-0.8B` as downloaded 2026-11-26 to `/mnt/ai/models/Qwen3.5-0.8B` (weights file hash in the result), loaded with `AutoModelForCausalLM` (text model only), fp32, one GPU, one prompt at a time, greedy decoding with the library's cache.

**Prompt.** The model's own chat template, one user message, generation prompt added, `enable_thinking=False`. At most 512 new tokens; generation ends at the model's end token. `hit_limit` = 512 tokens produced without an end token.

**Answer.** The text of the first fenced code block (```` ```python ```` or bare ```` ``` ````; if the fence is never closed, everything after it). If there is no fence, the whole reply.

**Problems.** Splits as in spec 005: practice = MBPP task_id 511 to 974 (464); held-out = MBPP 11 to 510 and HumanEval, not touched here.
- MBPP user message: `{text}\nYour code should pass this test:\n{first test}`. Program run: `test_setup_code`, the answer, then all the problem's tests.

**Sandbox and pass.** As spec 005 (`common.run_sandboxed`, self-check first, 10 s, 1 GB, no network, only `/usr`). Pass = exit code 0. Failure types as spec 005 (`common.classify`), applied to the extracted answer.

**Maths check.** GSM8K train questions 1 to 300 (practice). User message: the question followed by `\nGive the final answer as a number on the last line.` Solved = the last number in the reply (commas removed) equals the reference answer after `####`, compared as numbers. Also recorded: replies that hit the token limit.

**Go / no-go.** The edit run goes ahead if the practice pass rate is between 5% and 80%. Outside that: stop and report.

**Repeat.** `run_baseline.sh` runs the baseline twice; result files must be identical.
