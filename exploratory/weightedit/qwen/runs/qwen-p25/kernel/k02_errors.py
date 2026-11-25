"""No model pass: after cutting the model's own tests, what stops the remaining answers (work set)?"""
import json, re, sys, collections
sys.path.insert(0, "exploratory/weightedit/qwen")
import qcommon as Q
C = Q.C
exec(open("exploratory/weightedit/qwen/runs/qwen-p25/kernel/k01_counterfactual.py").read().split("tab = ")[0].split('"""', 2)[2])
tab, ex = collections.Counter(), {}
for a in A:
    p = P[a["task_id"]]
    code = cut(Q.extract(a["reply"]))
    s, e = C.run_sandboxed(C.mbpp_program(p, code))
    if s == "pass": k = "pass"
    elif s == "timeout": k = "timeout"
    else:
        last = e.strip().split("\n")[-1]
        k = last.split(":")[0][:40]
        if k == "NameError":
            want = re.match(r"assert\s+\(*\s*([A-Za-z_0-9]+)", p["test_list"][0])
            k = "NameError:function_name" if want and f"def {want.group(1)}(" not in code else "NameError:other"
        if k == "SyntaxError": k = "SyntaxError(cut)" if a["hit_limit"] else "SyntaxError"
    tab[k] += 1; ex.setdefault(k, []).append(a["task_id"])
for k, v in tab.most_common(): print(v, k, ex[k][:6])
