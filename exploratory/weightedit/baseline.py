"""Baseline per specs/005-weight-edit-coding/baseline-protocol.md.
usage: baseline.py <out.json> [n_problems]"""
import collections, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None
practice, noncode = C.load_practice()[:n], C.load_noncode()[:n]
C.sandbox_selfcheck()
res = {}
for name in C.MODELS:
    m = C.load(name)
    A, B = [], []
    for p in practice:
        comp, hit = C.generate(m, C.mbpp_prompt(p))
        status, err = C.run_sandboxed(C.mbpp_program(p, comp))
        A.append(dict(task_id=p["task_id"], outcome=C.classify(comp, hit, status, err), completion=comp))
        text, k = C.mbpp_code_text(p)
        s = C.score(m, text, k)
        s.update(task_id=p["task_id"])
        B.append(s)
    N = [C.score(m, q["text"]) for q in noncode]
    sc = [b for b in B if b["n"]]
    res[name] = dict(
        n_params=m["n_params"],
        weights_sha256=C.sha256(C.weights_path(name)),
        level_A=dict(pass_rate=sum(a["outcome"] == "pass" for a in A) / len(A), outcomes=dict(collections.Counter(a["outcome"] for a in A))),
        level_B=dict(mean_top1=sum(b["correct"] / b["n"] for b in sc) / len(sc),
                     bits_per_char=sum(b["nll"] for b in sc) / math.log(2) / sum(b["chars"] for b in sc)),
        noncode=dict(mean_top1=sum(x["correct"] / x["n"] for x in N) / len(N)),
        A=A, B=B,
    )
    print(name, json.dumps({k: res[name][k] for k in ("n_params", "level_A", "level_B", "noncode")}), flush=True)
    del m
ok = lambda v: 0.05 <= v <= 0.60
qa = [k for k, v in res.items() if ok(v["level_A"]["pass_rate"])]
qb = [k for k, v in res.items() if ok(v["level_B"]["mean_top1"])]
level, q = ("A", qa) if qa else ("B", qb)
res["_selection"] = dict(level=level if q else None, qualifying=q, chosen=min(q, key=lambda k: res[k]["n_params"]) if q else None)
print("SELECTION", res["_selection"], flush=True)
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
