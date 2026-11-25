"""Download MBPP, HumanEval and GSM8K train questions; write the splits and their checksums."""
import gzip, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, REPO, sha256

URLS = {
    "mbpp.jsonl": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
    "HumanEval.jsonl.gz": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
    "gsm8k_train.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
}
raw, out = f"{DATA}/raw", f"{DATA}/data"
os.makedirs(raw, exist_ok=True); os.makedirs(out, exist_ok=True)
for name, url in URLS.items():
    if not os.path.exists(f"{raw}/{name}"):
        subprocess.run(["curl", "-sSfL", "-o", f"{raw}/{name}", url], check=True)

def dump(name, rows):
    with open(f"{out}/{name}", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)

mbpp = [json.loads(l) for l in open(f"{raw}/mbpp.jsonl")]
for p in mbpp:
    p["code"] = p["code"].replace("\r\n", "\n").replace("\r", "\n")
counts = {
    "practice_mbpp.jsonl": dump("practice_mbpp.jsonl", [p for p in mbpp if 511 <= p["task_id"] <= 974]),
    "heldout_mbpp.jsonl": dump("heldout_mbpp.jsonl", [p for p in mbpp if 11 <= p["task_id"] <= 510]),
    "heldout_humaneval.jsonl": dump("heldout_humaneval.jsonl", [json.loads(l) for l in gzip.open(f"{raw}/HumanEval.jsonl.gz", "rt")]),
    "noncode_gsm8k.jsonl": dump("noncode_gsm8k.jsonl", [dict(text=json.loads(l)["question"]) for l in open(f"{raw}/gsm8k_train.jsonl")][:300]),
}
info = dict(raw={n: sha256(f"{raw}/{n}") for n in URLS}, splits={n: dict(rows=c, sha256=sha256(f"{out}/{n}")) for n, c in counts.items()})
json.dump(info, open(f"{REPO}/exploratory/weightedit/results/data-checksums.json", "w"), indent=1)
print(json.dumps(info, indent=1))
