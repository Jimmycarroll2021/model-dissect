"""Pass 5 (read, original): same reply text under the plain prompt and under the prompt with an instruction.
128 work problems x 2 prompts. Reply = baseline reply up to (not including) the model's own tests, at most 700 chars."""
import json, re, sys
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
Q, C = H.Q, H.C
INS2 = "\nName the function exactly as it is called in the test. Do not write tests or example calls after the function."
r = {a["task_id"]: a for a in json.load(open(f"{C.DATA}/runs/qwen-baseline-run1.json"))["A"]}
P = [p for p in Q.load_practice() if p["task_id"] <= 800 and "def " in r[p["task_id"]]["reply"]][:128]
def head(t):
    L, seen, out = t.split("\n"), False, []
    for l in L:
        if l.startswith("def "): seen = True
        if seen and re.match(r"(#|assert|print\(|if __name__|```)", l): break
        out.append(l)
    return "\n".join(out)[:700]
T = []
for p in P:
    rep = head(r[p["task_id"]]["reply"])
    T += [dict(user=Q.mbpp_user(p), reply=rep), dict(user=Q.mbpp_user(p) + INS2, reply=rep)]
keys = ["resid_pre7", "resid_pre11", "resid_pre15", "mlp_in6", "mlp_in10", "mlp_in14", "mlp_act6", "mlp_act10", "mlp_act14"]
print(H.read_pass(T, keys, "original"), len(T))
