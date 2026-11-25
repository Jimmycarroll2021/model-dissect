"""Held-out evaluation per specs/006-qwen-weight-edit-coding/edit-protocol.md. The only file that reads held-out data.
usage: qevaluate.py gen <run_dir> <unedited|edited|random> <out.json>     (one set of weights, all held-out sets)
       qevaluate.py verdict <unedited.json> <edited.json> <random.json> <out.json>"""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as Q
C = Q.C
SEED = 20261126


def humaneval_user(p):
    return f'Complete this Python function. Reply with the full function in one code block.\n```python\n{p["prompt"]}```'


def run_humaneval(m, p):
    text, hit = Q.reply(m, humaneval_user(p))
    code = Q.extract(text)
    status, err = C.run_sandboxed(p["prompt"] + "\n" + code + "\n" + p["test"] + f"\ncheck({p['entry_point']})\n")
    return dict(task_id=p["task_id"], outcome=C.classify(code, hit, status, err), hit_limit=hit, reply=text)


def gen(run, which, out):
    import qharness as H
    assert json.load(open(f"{run}/state.json"))["frozen"], "edit is not frozen"
    jl = lambda n: [json.loads(l) for l in open(f"{C.DATA}/heldout/{n}")]
    C.sandbox_selfcheck()
    E = H.apply_edit(run)
    if which == "random":  # same changes in value, placed at randomly permuted positions among the changed entries
        rng, R = np.random.Generator(np.random.PCG64(SEED)), {}
        for name, t in E.items():
            o = H.original(name).flatten()
            idx = torch.nonzero(t.flatten() != o)[:, 0]
            r = o.clone()
            r[idx] = o[idx] + (t.flatten() - o)[idx][torch.from_numpy(rng.permutation(len(idx)))]
            R[name] = r.view(t.shape)
        E = R
    m = H.use(None if which == "unedited" else E)
    res = dict(which=which, edit={n: H.tsha(t) for n, t in E.items()} if which != "unedited" else {},
               mbpp=[Q.run_mbpp(m, p) for p in jl("heldout_mbpp.jsonl")],
               humaneval=[run_humaneval(m, p) for p in jl("heldout_humaneval.jsonl")],
               maths=[Q.run_maths(m, q) for q in jl("heldout_maths_gsm8k.jsonl")])
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print(which, {s: sum(ok(r) for r in res[s]) / len(res[s]) for s in ("mbpp", "humaneval", "maths")}, flush=True)


def ok(r):
    return bool(r["solved"]) if "solved" in r else r["outcome"] == "pass"


def diff(a, b):
    d = a - b
    g = np.random.Generator(np.random.PCG64(SEED))
    bs = d[g.integers(0, len(d), (10000, len(d)))].mean(1)
    return dict(mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def verdict(files, out):
    R = {w: json.load(open(f)) for w, f in zip(("unedited", "edited", "random"), files)}
    assert all(R[w]["which"] == w for w in R)
    S = {}
    for s in ("mbpp", "humaneval", "maths"):
        v = {w: np.array([ok(r) for r in R[w][s]], float) for w in R}
        S[s] = dict(n=len(v["unedited"]), **{w: float(v[w].mean()) for w in v},
                    hit_limit={w: sum(r["hit_limit"] for r in R[w][s]) for w in R},
                    edited_minus_unedited=diff(v["edited"], v["unedited"]), edited_minus_random=diff(v["edited"], v["random"]))
        if s != "maths":
            S[s]["outcomes"] = {w: {k: sum(r["outcome"] == k for r in R[w][s]) for k in sorted({r["outcome"] for r in R[w][s]})} for w in R}
    gain = [s for s in ("mbpp", "humaneval") if S[s]["edited_minus_unedited"]["mean"] >= 0.03
            and S[s]["edited_minus_unedited"]["ci95"][0] > 0 and S[s]["edited_minus_random"]["mean"] >= 0.03]
    side = S["maths"]["edited_minus_unedited"]["mean"] >= -0.02
    res = dict(edit=R["edited"]["edit"], sets=S, verdict=dict(gain_on=gain, side_effect_ok=bool(side), result="GAIN" if gain and side else "NO_GAIN"))
    json.dump(res, open(out, "w"), indent=1)
    for s in S:
        print(s, json.dumps(S[s]))
    print("VERDICT", res["verdict"])


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(*sys.argv[2:5])
    else:
        verdict(sys.argv[2:5], sys.argv[5])
