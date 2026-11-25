"""Download MBPP, HumanEval and GSM8K train questions (checked against pinned hashes); write the splits and their checksums.
Practice files go to DATA/practice; held-out files go to DATA/heldout, which only evaluate.py reads."""
import gzip, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, REPO, sha256

SRC = {  # name: (url, sha256)
    "mbpp.jsonl": ("https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl", "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"),
    "HumanEval.jsonl.gz": ("https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz", "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"),
    "gsm8k_train.jsonl": ("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl", "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"),
}
raw, res = f"{DATA}/raw", f"{REPO}/exploratory/weightedit/results"
for d in (raw, res, f"{DATA}/practice", f"{DATA}/heldout"):
    os.makedirs(d, exist_ok=True)
for name, (url, want) in SRC.items():
    if not os.path.exists(f"{raw}/{name}"):
        subprocess.run(["curl", "-sSfL", "-o", f"{raw}/{name}.part", url], check=True)
        os.replace(f"{raw}/{name}.part", f"{raw}/{name}")
    got = sha256(f"{raw}/{name}")
    assert got == want, f"{name}: sha256 {got} does not match the pinned {want}"

def dump(path, rows):
    with open(f"{DATA}/{path}", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dict(rows=len(rows), sha256=sha256(f"{DATA}/{path}"))

mbpp = [json.loads(l) for l in open(f"{raw}/mbpp.jsonl")]
for p in mbpp:
    p["code"] = p["code"].replace("\r\n", "\n").replace("\r", "\n")
splits = {
    "practice/practice_mbpp.jsonl": dump("practice/practice_mbpp.jsonl", [p for p in mbpp if 511 <= p["task_id"] <= 974]),
    "practice/noncode_gsm8k.jsonl": dump("practice/noncode_gsm8k.jsonl", [dict(text=json.loads(l)["question"]) for l in open(f"{raw}/gsm8k_train.jsonl")][:300]),
    "heldout/heldout_mbpp.jsonl": dump("heldout/heldout_mbpp.jsonl", [p for p in mbpp if 11 <= p["task_id"] <= 510]),
    "heldout/heldout_noncode_gsm8k.jsonl": dump("heldout/heldout_noncode_gsm8k.jsonl", [dict(text=json.loads(l)["question"]) for l in open(f"{raw}/gsm8k_train.jsonl")][1000:1300]),
    "heldout/heldout_humaneval.jsonl": dump("heldout/heldout_humaneval.jsonl", [json.loads(l) for l in gzip.open(f"{raw}/HumanEval.jsonl.gz", "rt")]),
}
info = dict(raw={n: s for n, (u, s) in SRC.items()}, splits=splits)
json.dump(info, open(f"{res}/data-checksums.json", "w"), indent=1)
print(json.dumps(info, indent=1))
