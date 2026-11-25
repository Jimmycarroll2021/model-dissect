"""Pass 4 (score, original weights): the first 200 work problems with a plain instruction added to the prompt.
Asks: would the model do the right thing if told? (If yes, the instruction's effect is what an edit should install.)"""
import json, sys, collections
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
Q, C = H.Q, H.C
INS = "\nName the function exactly as it is called in the test. Reply with only the function in one code block: no tests, no examples, no explanation."
H.clear_edit()
P = [p for p in Q.load_practice() if p["task_id"] <= 800][:200]
summ, res = H.score_pass([("custom", Q.mbpp_user(p) + INS) for p in P], "original")
base = {a["task_id"]: a["outcome"] for a in json.load(open(f"{C.DATA}/runs/qwen-baseline-run1.json"))["A"]}
out = []
for p, r in zip(P, res["original"]):
    code = Q.extract(r["reply"])
    s, e = C.run_sandboxed(C.mbpp_program(p, code))
    out.append(C.classify(code, r["hit_limit"], s, e))
print("with instruction:", dict(collections.Counter(out)), "| hit limit", sum(r["hit_limit"] for r in res["original"]))
print("baseline same 200:", dict(collections.Counter(base[p["task_id"]] for p in P)))
print("mean reply chars", sum(len(r["reply"]) for r in res["original"]) / 200)
print(res["original"][1]["reply"][:500])
