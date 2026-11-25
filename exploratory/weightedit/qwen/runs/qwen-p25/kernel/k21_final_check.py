"""Final check (allowed once): the 174 held-back practice problems (task_id > 800) and 26 maths questions (100..125), edited weights."""
import json, re, sys, collections
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
Q, C = H.Q, H.C
base = json.load(open(f"{C.DATA}/runs/qwen-baseline-run1.json"))
P = [p for p in Q.load_practice() if p["task_id"] > 800]
E_ = H.current_edit()
for L in (14, 23):
    G = E_[f"model.layers.{L}.mlp.gate_proj.weight"]; print("layer", L, "neuron", int((G != H.original(f"model.layers.{L}.mlp.gate_proj.weight")).any(1).nonzero()[0]))
summ, res = H.score_pass([("mbpp", p["task_id"]) for p in P] + [("maths", i) for i in range(100, 126)], "edited", final_check=True)
B = {x["task_id"]: x for x in base["A"]}; E = res["edited"][:len(P)]
print("held-back baseline:", dict(collections.Counter(B[p["task_id"]]["outcome"] for p in P)))
print("held-back edited:  ", dict(collections.Counter(r["outcome"] for r in E)))
g = sum(r["outcome"] == "pass" and B[p["task_id"]]["outcome"] != "pass" for p, r in zip(P, E)); l = sum(r["outcome"] != "pass" and B[p["task_id"]]["outcome"] == "pass" for p, r in zip(P, E))
print("gained", g, "lost", l, "of", len(P), "| hit limit", sum(B[p["task_id"]]["hit_limit"] for p in P), "->", sum(r["hit_limit"] for r in E))
M = res["edited"][len(P):]
print("maths baseline", sum(x["solved"] for x in base["M"][100:126]), "edited", sum(r["solved"] for r in M), "of 26; replies changed", sum(x["reply"] != r["reply"] for x, r in zip(base["M"][100:126], M)))
