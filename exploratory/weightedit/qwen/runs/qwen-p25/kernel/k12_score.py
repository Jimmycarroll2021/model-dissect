"""Score pass (edited only) on work problems [a:b) and maths [0:n); compare with the baseline's replies to the same items.
usage: k12_score.py a b n_maths"""
import json, re, sys, collections
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
Q, C = H.Q, H.C
a, b, nm = (int(x) for x in sys.argv[1:4])
base = json.load(open(f"{C.DATA}/runs/qwen-baseline-run1.json"))
P = [p for p in Q.load_practice() if p["task_id"] <= 800][a:b]
summ, res = H.score_pass([("mbpp", p["task_id"]) for p in P] + [("maths", i) for i in range(nm)], "edited")
B = {x["task_id"]: x for x in base["A"]}
E = res["edited"][:len(P)]
def wrong(p, r):
    want = re.match(r"assert\s+\(*\s*([A-Za-z_0-9]+)", p["test_list"][0]).group(1)
    d = re.findall(r"^def ([A-Za-z_0-9]+)\(", Q.extract(r["reply"]), re.M)
    return bool(d) and want not in d
print("edit:", list(H.current_edit()))
print("code  baseline:", dict(collections.Counter(B[p["task_id"]]["outcome"] for p in P)), "wrong names", sum(wrong(p, B[p["task_id"]]) for p in P))
print("code  edited:  ", dict(collections.Counter(r["outcome"] for r in E)), "wrong names", sum(wrong(p, r) for p, r in zip(P, E)))
g = sum(r["outcome"] == "pass" and B[p["task_id"]]["outcome"] != "pass" for p, r in zip(P, E)); l = sum(r["outcome"] != "pass" and B[p["task_id"]]["outcome"] == "pass" for p, r in zip(P, E))
print("gained", g, "lost", l, "of", len(P), "| hit limit", sum(B[p["task_id"]]["hit_limit"] for p in P), "->", sum(r["hit_limit"] for r in E))
if nm: print("maths baseline", sum(x["solved"] for x in base["M"][:nm]), "edited", sum(r["solved"] for r in res["edited"][len(P):]), "of", nm)
print(E[1]["reply"][:600])
