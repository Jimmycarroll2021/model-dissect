"""No model pass: re-judge the baseline replies (work set, task_id <= 800) with the model's own trailing tests cut off.
Cut = first top-level line after the first def that starts with '#', 'assert', 'print(' or 'if __name__'."""
import json, re, sys, collections
sys.path.insert(0, "exploratory/weightedit/qwen")
import qcommon as Q
C = Q.C
r = json.load(open(f"{C.DATA}/runs/qwen-baseline-run1.json"))
P = {p["task_id"]: p for p in Q.load_practice()}
A = [a for a in r["A"] if a["task_id"] <= 800]
def cut(code):
    L, seen, out = code.split("\n"), False, []
    for l in L:
        if l.startswith("def "): seen = True
        if seen and re.match(r"(#|assert|print\(|if __name__)", l): break
        out.append(l)
    return "\n".join(out)
tab = collections.Counter()
for a in A:
    code = cut(Q.extract(a["reply"]))
    s, e = C.run_sandboxed(C.mbpp_program(P[a["task_id"]], code))
    tab[(a["outcome"], s)] += 1
for k in sorted(tab): print(k, tab[k])
print("pass before", sum(a["outcome"] == "pass" for a in A), "after cut", sum(v for (o, s), v in tab.items() if s == "pass"), "of", len(A))
