"""Pass 2 (read): prompt + the model's own reply up to its first 'def ', then the name the test uses and '('.
Work problems only (task_id <= 800), first 256 that have a 'def ' in the baseline reply."""
import json, re, sys
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
Q, C = H.Q, H.C
r = json.load(open(f"{C.DATA}/runs/qwen-baseline-run1.json"))
P = {p["task_id"]: p for p in Q.load_practice()}
T, meta = [], []
for a in r["A"]:
    if a["task_id"] > 800: continue
    k = re.search(r"^def ([A-Za-z_0-9]+)\(", a["reply"], re.M)
    if not k: continue
    want = re.match(r"assert\s+\(*\s*([A-Za-z_0-9]+)", P[a["task_id"]]["test_list"][0]).group(1)
    T.append(dict(user=Q.mbpp_user(P[a["task_id"]]), reply=a["reply"][:k.start(1)] + want + "("))
    meta.append(dict(task_id=a["task_id"], own=k.group(1), want=want))
T, meta = T[:256], meta[:256]
keys = [f"attn_z{L}" for L in range(24)] + ["resid_final"]
path = H.read_pass(T, keys, "original", reply_only=False)
json.dump(meta, open(path.replace(".npz", "_meta.json"), "w"))
print(path, len(T), sum(m["own"] != m["want"] for m in meta), "wrong names")
