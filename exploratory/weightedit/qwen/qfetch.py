"""Adds the maths splits with answers (raw file already hash-checked by ../fetch_data.py, re-checked here)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA, REPO, sha256

raw = f"{DATA}/raw/gsm8k_train.jsonl"
assert sha256(raw) == "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
rows = [json.loads(l) for l in open(raw)]
out = {}
for path, part in (("practice/maths_gsm8k.jsonl", rows[:300]), ("heldout/heldout_maths_gsm8k.jsonl", rows[1000:1300])):
    with open(f"{DATA}/{path}", "w") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out[path] = dict(rows=len(part), sha256=sha256(f"{DATA}/{path}"))
json.dump(out, open(f"{REPO}/exploratory/weightedit/qwen/results/data-checksums.json", "w"), indent=1)
print(json.dumps(out, indent=1))
