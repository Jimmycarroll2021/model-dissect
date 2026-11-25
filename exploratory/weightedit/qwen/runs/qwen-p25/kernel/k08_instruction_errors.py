"""No pass: what still stops the instructed replies of pass 4, and the rename upper bound."""
import json, re, sys, collections
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
Q, C = H.Q, H.C
P = [p for p in Q.load_practice() if p["task_id"] <= 800][:200]
R = json.load(open(f"{H.RUN}/requests/fwd_4.json"))["results"]["original"]
tab, wrong, ren_pass, n_pass = collections.Counter(), 0, 0, 0
for p, r in zip(P, R):
    code = Q.extract(r["reply"])
    want = re.match(r"assert\s+\(*\s*([A-Za-z_0-9]+)", p["test_list"][0]).group(1)
    defs = re.findall(r"^def ([A-Za-z_0-9]+)\(", code, re.M)
    bad = bool(defs) and want not in defs
    wrong += bad
    s, e = C.run_sandboxed(C.mbpp_program(p, code))
    n_pass += s == "pass"
    tab["pass" if s == "pass" else e.strip().split("\n")[-1].split(":")[0][:30] if s == "fail" else s] += 1
    if bad: code = re.sub(rf"\b{defs[-1]}\b", want, code)
    ren_pass += C.run_sandboxed(C.mbpp_program(p, code))[0] == "pass"
print(tab.most_common()); print("wrong names", wrong, "| pass", n_pass, "| pass if renamed", ren_pass, "of 200")
