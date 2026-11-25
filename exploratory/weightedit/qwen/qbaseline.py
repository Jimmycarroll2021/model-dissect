"""Baseline per specs/006-qwen-weight-edit-coding/baseline-protocol.md.
usage: qbaseline.py <out.json> [n_problems]"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as Q
C = Q.C

out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None
practice, maths = Q.load_practice()[:n], Q.load_maths()[:n]
C.sandbox_selfcheck()
m = Q.load()
A = [Q.run_mbpp(m, p) for p in practice]
M = [Q.run_maths(m, q) for q in maths]
rate = sum(a["outcome"] == "pass" for a in A) / len(A)
res = dict(
    weights_sha256=C.sha256(Q.WEIGHTS),
    n_params=sum(p.numel() for p in m["model"].parameters()),
    code=dict(n=len(A), pass_rate=rate, outcomes=dict(collections.Counter(a["outcome"] for a in A)), hit_limit=sum(a["hit_limit"] for a in A)),
    maths=dict(n=len(M), solve_rate=sum(x["solved"] for x in M) / len(M), hit_limit=sum(x["hit_limit"] for x in M)),
    go=bool(0.05 <= rate <= 0.80),
    A=A, M=M,
)
print("RESULT", json.dumps({k: res[k] for k in ("n_params", "code", "maths", "go")}), flush=True)
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
