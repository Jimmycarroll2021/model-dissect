"""Shared pieces for specs/005-weight-edit-coding: models, tokenisation, greedy generation, scoring, sandbox.
Held-out problems are NOT readable from here; only evaluate.py knows their paths."""

import collections, hashlib, json, os, subprocess, sys, tempfile, torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = "/mnt/ai/model-dissect-data/weightedit"
W = "/mnt/ai/workspaces"
MODELS = {
    "minimind": f"{W}/minimind-atlas-001-20261118/artifacts/original",
    "pythia-14m": f"{W}/model-transfer-002-pythia-14m/artifacts/original",
    "gpt2": f"{W}/model-transfer-003-gpt2/artifacts/original",
}
STOPS = ["\nclass ", "\nassert", '\n"""', "\nprint(", "\nif __name__", "\n#"]
MAX_NEW = 256
torch.set_grad_enabled(False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def weights_path(name):
    return os.path.join(MODELS[name], "full_sft_768.pth" if name == "minimind" else "model.safetensors")


def load(name, dev="cuda"):
    """Returns dict(model, tok, bos, ctx, n_params). fp32, eval."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

    d = MODELS[name]
    if name == "minimind":
        sys.path.insert(0, REPO)
        from oracle.vendor.model_minimind import MiniMindConfig, MiniMindForCausalLM

        model = MiniMindForCausalLM(MiniMindConfig(flash_attn=False)).float()
        model.load_state_dict(torch.load(weights_path(name), map_location="cpu", weights_only=True), strict=True)
        tok = PreTrainedTokenizerFast(tokenizer_file=f"{d}/tokenizer.json")
        bos, ctx, eos = 1, 2048, 2
    else:
        model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32)
        tok = AutoTokenizer.from_pretrained(d)
        bos = model.config.bos_token_id
        ctx = 1024 if name == "gpt2" else 2048
        eos = model.config.eos_token_id
    model = model.to(dev).eval()
    n = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    return dict(name=name, model=model, tok=tok, bos=bos, eos=eos, ctx=ctx, n_params=n, dev=dev)


def encode(m, text):
    """ids (with start token) and per-token start character offset (-1 for the start token)."""
    enc = m["tok"](text, return_offsets_mapping=True, add_special_tokens=False)
    ids = [m["bos"]] + enc["input_ids"]
    starts = [-1] + [a for a, b in enc["offset_mapping"]]
    m["_ends"] = ([0] + [b for a, b in enc["offset_mapping"]])[: m["ctx"]]
    return ids[: m["ctx"]], starts[: m["ctx"]]


def logits_of(m, ids):
    x = torch.tensor([ids], device=m["dev"])
    return m["model"](input_ids=x).logits[0]


def score(m, text, from_char=0):
    """Teacher-forced top-1 accuracy and summed NLL (nats) over tokens starting at or after from_char."""
    ids, starts = encode(m, text)
    lg = logits_of(m, ids)[:-1].float()
    tgt = torch.tensor(ids[1:], device=m["dev"])
    mask = torch.tensor([s >= from_char for s in starts[1:]], device=m["dev"])
    n = int(mask.sum())
    chars = max(m["_ends"][-1] - from_char, 0)  # code characters actually covered (after truncation)
    if n == 0:
        return dict(n=0, correct=0, nll=0.0, chars=0)
    correct = int(((lg.argmax(-1) == tgt) & mask).sum())
    nll = float(torch.nn.functional.cross_entropy(lg[mask], tgt[mask], reduction="sum"))
    return dict(n=n, correct=correct, nll=nll, chars=chars)


def generate(m, prompt, max_new=MAX_NEW, stops=STOPS):
    """Greedy, full forward each step (no cache). Returns (completion text cut at the first stop, hit_limit)."""
    ids, _ = encode(m, prompt)
    n0 = len(ids)
    for _ in range(max_new):
        if len(ids) >= m["ctx"]:
            break
        nxt = int(logits_of(m, ids)[-1].argmax())
        if nxt == m["eos"]:  # the model ended its answer
            return m["tok"].decode(ids[n0:]), False
        ids.append(nxt)
        text = m["tok"].decode(ids[n0:])
        cut = min((text.find(s) for s in stops if s in text), default=-1)
        if cut >= 0:
            return text[:cut], False
    return m["tok"].decode(ids[n0:]), True


def run_sandboxed(program, timeout=10):
    """Runs a Python program with no network, only /usr visible (read-only), 1 GB memory, 1 MB of stderr kept.
    Returns (status, stderr tail). Raises if the sandbox itself fails to start."""
    with tempfile.TemporaryDirectory(dir="/dev/shm") as d:
        os.mkdir(f"{d}/in"); os.mkdir(f"{d}/out")
        with open(f"{d}/in/prog.py", "w") as f:
            f.write(program)
        cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
               "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--dev", "/dev", "--proc", "/proc",
               "--tmpfs", "/tmp", "--dir", "/tmp/work", "--ro-bind", f"{d}/in", "/tmp/work", "--dir", "/tmp/out",
               "--bind", f"{d}/out", "/tmp/out", "--chdir", "/tmp", "--clearenv", "--setenv", "PATH", "/usr/bin",
               "bash", "-c", "ulimit -v 1048576; ulimit -f 1024; exec /usr/bin/python3.12 -I /tmp/work/prog.py >/dev/null 2>/tmp/out/err"]
        try:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return "timeout", ""
        if r.stderr.startswith("bwrap:") or not os.path.exists(f"{d}/out/err"):
            raise RuntimeError(f"sandbox did not start: {r.stderr[:300]}")
        with open(f"{d}/out/err", "rb") as f:
            f.seek(max(os.path.getsize(f"{d}/out/err") - 400, 0))
            err = f.read().decode("utf-8", "replace")
        return ("pass" if r.returncode == 0 else "fail"), err


def sandbox_selfcheck():
    """Abort unless the sandbox passes a good program, fails a bad one for the right reason, and has no network or home."""
    assert run_sandboxed("assert 1 + 1 == 2\n")[0] == "pass"
    s, e = run_sandboxed("assert 1 + 1 == 3\n")
    assert s == "fail" and "AssertionError" in e, (s, e)
    probe = "import socket, os\nassert not os.path.exists('/home') and not os.path.exists('/mnt')\ntry:\n    socket.create_connection(('1.1.1.1', 53), 2)\nexcept OSError:\n    raise SystemExit(0)\nraise SystemExit(1)\n"
    assert run_sandboxed(probe)[0] == "pass", "sandbox can see /home or the network"
    assert run_sandboxed("while True: pass\n", timeout=2)[0] == "timeout"


def classify(completion, hit_limit, status, stderr):
    if status == "pass":
        return "pass"
    if not completion.strip() or "def " not in completion:
        return "no_code"
    lines = collections.Counter(l for l in completion.split("\n") if l.strip())
    if lines and max(lines.values()) >= 4:
        return "repetition_loop"
    if hit_limit:
        return "cut_at_limit"
    try:
        compile(completion, "<completion>", "exec")
    except (SyntaxError, ValueError):
        return "syntax_error"
    if status == "timeout":
        return "timeout"
    return "wrong_answer" if "AssertionError" in stderr else "runtime_error"


def mbpp_prompt(p):
    return f'"""\n{p["text"]}\n{p["test_list"][0]}\n"""\n'


def mbpp_program(p, completion):
    return "\n".join([p.get("test_setup_code", ""), completion, ""] + p["test_list"]) + "\n"


def mbpp_code_text(p):
    head = f'# {p["text"]}\n'
    return head + p["code"], len(head)


def _jsonl(path):
    return [json.loads(l) for l in open(path)]


def load_practice():
    return _jsonl(f"{DATA}/practice/practice_mbpp.jsonl")


def load_noncode():
    return _jsonl(f"{DATA}/practice/noncode_gsm8k.jsonl")
