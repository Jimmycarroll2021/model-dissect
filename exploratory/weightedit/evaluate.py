"""Held-out evaluation per specs/005-weight-edit-coding/edit-protocol.md. The only file that reads held-out data.
usage: evaluate.py <run_dir> <out.json>"""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, harness as H

run, out = sys.argv[1], sys.argv[2]
assert json.load(open(f"{run}/state.json"))["frozen"], "edit is not frozen"
HELD = f"{C.DATA}/heldout"
jl = lambda n: [json.loads(l) for l in open(f"{HELD}/{n}")]
SETS = {
    "mbpp": [C.mbpp_code_text(p) for p in jl("heldout_mbpp.jsonl")],
    "humaneval": [(p["prompt"] + p["canonical_solution"], len(p["prompt"])) for p in jl("heldout_humaneval.jsonl")],
    "noncode": [(q["text"], 0) for q in jl("heldout_noncode_gsm8k.jsonl")],
}
O, E = H.original(), H.apply_edit(run)
rng = np.random.Generator(np.random.PCG64(20261125))
Rnd = {}
for name, t in E.items():
    o = O[name].flatten()
    idx = torch.nonzero(t.flatten() != o)[:, 0]
    delta = (t.flatten() - o)[idx]
    r = o.clone()
    r[idx] = o[idx] + delta[torch.from_numpy(rng.permutation(len(idx)))]
    Rnd[name] = r.view(t.shape)
res = dict(edit={n: H.tsha(t) for n, t in E.items()}, sets={})
acc = {}
for w, edit in (("unedited", None), ("edited", E), ("random", Rnd)):
    m = H.build(edit)
    for s, items in SETS.items():
        r = [C.score(m, t, k) for t, k in items]
        acc[(s, w)] = np.array([x["correct"] / x["n"] if x["n"] else np.nan for x in r])
    del m

def diff(a, b):
    d = (a - b)[~np.isnan(a - b)]
    g = np.random.Generator(np.random.PCG64(20261125))
    bs = d[g.integers(0, len(d), (10000, len(d)))].mean(1)
    return dict(mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])

for s in SETS:
    res["sets"][s] = dict(
        n=int((~np.isnan(acc[(s, "unedited")])).sum()),
        **{w: float(np.nanmean(acc[(s, w)])) for w in ("unedited", "edited", "random")},
        edited_minus_unedited=diff(acc[(s, "edited")], acc[(s, "unedited")]),
        edited_minus_random=diff(acc[(s, "edited")], acc[(s, "random")]),
        per_problem={w: [None if np.isnan(x) else float(x) for x in acc[(s, w)]] for w in ("unedited", "edited", "random")},
    )
S = res["sets"]
code_gain = [s for s in ("mbpp", "humaneval")
             if S[s]["edited_minus_unedited"]["mean"] >= 0.03 and S[s]["edited_minus_unedited"]["ci95"][0] > 0 and S[s]["edited_minus_random"]["mean"] >= 0.03]
side_ok = S["noncode"]["edited_minus_unedited"]["mean"] >= -0.02
res["verdict"] = dict(gain_on=code_gain, side_effect_ok=bool(side_ok), result="GAIN" if code_gain and side_ok else "NO_GAIN")
json.dump(res, open(out, "w"), indent=1)
for s in SETS:
    print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in S[s].items() if k != "per_problem"})
print("VERDICT", res["verdict"])
