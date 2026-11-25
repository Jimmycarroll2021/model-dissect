"""No model pass: upper bound if the function carried the name used in the given test (work set).
The answer's last top-level def is renamed to the test's function name; own tests cut as in k01."""
import json, re, sys, collections
sys.path.insert(0, "exploratory/weightedit/qwen")
import qcommon as Q
C = Q.C
exec(open("exploratory/weightedit/qwen/runs/qwen-p25/kernel/k01_counterfactual.py").read().split("tab = ")[0].split('"""', 2)[2])
n = dict(base=0, cut=0, cut_rename=0, rename_only=0)
wrong, first_tok = 0, collections.Counter()
for a in A:
    p = P[a["task_id"]]
    want = re.match(r"assert\s+\(*\s*([A-Za-z_0-9]+)", p["test_list"][0]).group(1)
    full = Q.extract(a["reply"])
    def ren(code):
        defs = re.findall(r"^def ([A-Za-z_0-9]+)\(", code, re.M)
        if not defs or want in defs: return code
        return re.sub(rf"\b{defs[-1]}\b", want, code)
    defs = re.findall(r"^def ([A-Za-z_0-9]+)\(", full, re.M)
    if defs and want not in defs: wrong += 1
    n["base"] += a["outcome"] == "pass"
    n["cut"] += C.run_sandboxed(C.mbpp_program(p, cut(full)))[0] == "pass"
    n["cut_rename"] += C.run_sandboxed(C.mbpp_program(p, ren(cut(full))))[0] == "pass"
    n["rename_only"] += C.run_sandboxed(C.mbpp_program(p, ren(full)))[0] == "pass"
print(n, "of", len(A), "| answers whose defs lack the test's name:", wrong)
