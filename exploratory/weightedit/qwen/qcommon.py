"""Shared pieces for specs/006-qwen-weight-edit-coding: model, chat prompt, greedy reply, answer extraction.
Held-out problems are NOT readable from here; only qevaluate.py knows their paths."""
import json, os, re, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C

QWEN = "/mnt/ai/models/Qwen3.5-0.8B"
WEIGHTS = f"{QWEN}/model.safetensors-00001-of-00001.safetensors"
MAX_NEW = 512
MATHS_TAIL = "\nGive the final answer as a number on the last line."
torch.set_grad_enabled(False)


def load(dev="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(QWEN)
    ends = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    assert all(isinstance(e, int) and e >= 0 for e in ends)
    return dict(model=model, tok=tok, dev=dev, ends=ends)


def chat_ids(m, user):
    p = m["tok"].apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return m["tok"](p, return_tensors="pt", add_special_tokens=False).input_ids.to(m["dev"])


def reply(m, user, max_new=MAX_NEW):
    """Greedy reply text and hit_limit (max_new tokens produced without an end token)."""
    x = chat_ids(m, user)
    o = m["model"].generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, do_sample=False,
                            temperature=None, top_p=None, top_k=None, eos_token_id=m["ends"], pad_token_id=m["ends"][1])
    new = o[0, x.shape[1]:].tolist()
    ended = bool(new) and new[-1] in m["ends"]
    return m["tok"].decode(new, skip_special_tokens=True), not ended


def extract(text):
    """First fenced code block; everything after an unclosed fence; else the whole reply."""
    k = re.search(r"```[A-Za-z0-9_+-]*[ \t]*\n", text)
    if not k:
        return text
    rest = text[k.end():]
    end = rest.find("```")
    return rest if end < 0 else rest[:end]


def mbpp_user(p):
    return f'{p["text"]}\nYour code should pass this test:\n{p["test_list"][0]}'


def run_mbpp(m, p):
    text, hit = reply(m, mbpp_user(p))
    code = extract(text)
    status, err = C.run_sandboxed(C.mbpp_program(p, code))
    return dict(task_id=p["task_id"], outcome=C.classify(code, hit, status, err), hit_limit=hit, reply=text)


def last_number(text):
    n = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    try:
        return float(n[-1].replace(",", "")) if n else None
    except ValueError:
        return None


def run_maths(m, q):
    text, hit = reply(m, q["question"] + MATHS_TAIL)
    got, want = last_number(text), float(q["answer"].split("####")[-1].strip().replace(",", ""))
    return dict(solved=got is not None and abs(got - want) < 1e-6, hit_limit=hit, reply=text)


def _jsonl(path):
    return [json.loads(l) for l in open(path)]


def load_practice():
    return _jsonl(f"{C.DATA}/practice/practice_mbpp.jsonl")


def load_maths():
    return _jsonl(f"{C.DATA}/practice/maths_gsm8k.jsonl")
