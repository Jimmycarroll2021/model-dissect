"""Like-for-like random control, per exploratory/control-v2-spec.md.
usage: control2.py <out.json>
Model, hooks and convention copied from audit2.py (which runs on import, so it is not imported)."""

import collections, glob, json, os, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = "/mnt/d/AI/cache/huggingface/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"
TC = (
    "/mnt/ai/models/gemma-scope-transcoders/layer_{}/width_16k/average_l0_{}/params.npz"
)
L0 = dict(
    enumerate(
        [
            76,
            65,
            49,
            54,
            88,
            87,
            95,
            70,
            52,
            72,
            88,
            5,
            6,
            8,
            8,
            8,
            10,
            12,
            13,
            12,
            11,
            13,
            15,
            25,
            37,
            41,
        ]
    )
)
VERDICTS = ("HOLDS", "WEAKER", "STRONGER", "NOT_DISTINCT", "WRONG_SIGN")
N_DRAWS, SEED0 = 64, 20262000
out_path = sys.argv[1]
dev = "cuda"

GRAPHS = {
    os.path.basename(p)[:-5]: p
    for d in ("graphs", "graphs2")
    for p in sorted(glob.glob(f"{HERE}/{d}/*.json"))
}
stored = [
    r
    for f in ("pilot-v1-run1.json", "scale-v1-run1.json")
    for r in json.load(open(f"{HERE}/results/{f}"))["edges"]
    if r.get("verdict") in VERDICTS
]
assert len(stored) == 75, len(stored)

tok = AutoTokenizer.from_pretrained(SNAP)
model = (
    AutoModelForCausalLM.from_pretrained(
        SNAP, torch_dtype=torch.float32, attn_implementation="eager"
    )
    .to(dev)
    .eval()
)
T = {}


def tc(L):
    if L not in T:
        P = np.load(TC.format(L, L0[L]))
        T[L] = {
            k: torch.tensor(P[k], device=dev)
            for k in ("W_enc", "W_dec", "b_enc", "threshold")
        }
    return T[L]


st = {"L": None, "edit": None, "pos": -1, "x": {}, "r": {}, "precap": None}


def mk(L):
    def pre(m, args):
        r = args[0][0, st["pos"]].detach()
        st["x"][L] = r * torch.rsqrt(r.pow(2).mean() + 1e-6)

    def post(m, args, out):
        if st["L"] != L or st["edit"] is None:
            return out
        out = out.clone()
        out[0, st["pos"]] += st["edit"]
        return out

    def resid(m, args, out):
        st["r"][L] = out[0][0, st["pos"]].detach()

    lay = model.model.layers[L]
    lay.pre_feedforward_layernorm.register_forward_pre_hook(pre)
    lay.post_feedforward_layernorm.register_forward_hook(post)
    lay.register_forward_hook(resid)


for L in L0:
    mk(L)
model.lm_head.register_forward_hook(
    lambda m, a, o: st.__setitem__("precap", o[0, -1].detach())
)


@torch.no_grad()
def precap(text, L=None, edit=None, pos=-1):
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    st["L"], st["edit"], st["pos"] = L, edit, pos
    model(ids)
    st["L"], st["edit"] = None, None
    return st["precap"].clone()


def cen(lg, t):
    return (lg[t] - lg.mean()).item()


def unit(v):
    return v / v.norm()


@torch.no_grad()
def run_edge(s):
    g = json.load(open(GRAPHS[s["graph"]]))
    text = g["metadata"]["prompt"].removeprefix("<bos>")
    n_tok = tok(text, return_tensors="pt").input_ids.shape[1]
    L, f, pos = (int(v) for v in s["node"].split("_"))
    p = pos - n_tok
    tgt = int(
        max(
            (n for n in g["nodes"] if n["feature_type"] == "logit"),
            key=lambda n: n["token_prob"],
        )["node_id"].split("_")[1]
    )
    Wd = tc(L)["W_dec"]
    base = cen(precap(text, pos=p), tgt)
    t = tc(L)
    pre = st["x"][L] @ t["W_enc"] + t["b_enc"]
    act = (pre * (pre > t["threshold"]))[f].item()
    r = st["r"][L].clone()
    v = act * Wd[f]
    E = base - cen(precap(text, L, -v, p), tgt)
    row = dict(
        graph=s["graph"],
        node=s["node"],
        W=s["W"],
        stored_E=s["E"],
        E=E,
        verdict_v1=s["verdict"],
    )
    row["gate_ok"] = bool(abs(E - s["E"]) <= 1e-3)
    if not row["gate_ok"]:
        return row
    rh, u = unit(r), unit(Wd[f])
    u2 = unit(u - (u @ rh) * rh)  # with rh, spans {rh, u}
    a = (v @ rh).item()
    b = max(v.norm().item() ** 2 - a * a, 0.0) ** 0.5
    target_norm = (r - v).norm().item()
    C, nerr = [], 0.0
    for i in range(N_DRAWS):
        rng = np.random.Generator(np.random.PCG64(SEED0 + i))
        w = torch.tensor(
            rng.standard_normal(r.shape[0]), dtype=torch.float32, device=dev
        )
        w = w - (w @ rh) * rh
        w = unit(w - (w @ u2) * u2)
        c = a * rh + b * w
        nerr = max(nerr, abs((r - c).norm().item() - target_norm))
        C.append(base - cen(precap(text, L, -c, p), tgt))
    n_ge = sum(x >= E for x in C)
    distinct = E > 0 and E - float(np.mean(C)) >= 0.25 and n_ge <= 3
    ratio = E / s["W"]
    v2 = (
        "WRONG_SIGN"
        if E <= 0
        else "NOT_DISTINCT"
        if not distinct
        else "WEAKER"
        if ratio < 0.5
        else "STRONGER"
        if ratio > 2
        else "HOLDS"
    )
    row.update(
        ratio=ratio,
        proj_frac=a / v.norm().item(),
        C_mean=float(np.mean(C)),
        C_max=float(np.max(C)),
        n_ge=n_ge,
        norm_check=nerr,
        verdict_v2=v2,
        C=C,
    )
    return row


rows = []
for s in stored:
    row = run_edge(s)
    rows.append(row)
    print(
        json.dumps(
            {
                k: (round(x, 3) if isinstance(x, float) else x)
                for k, x in row.items()
                if k != "C"
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
gate = all(r["gate_ok"] for r in rows)
res = dict(gate_ok=gate, edges=rows)
if gate:
    res["counts_v1"] = dict(collections.Counter(r["verdict_v1"] for r in rows))
    res["counts_v2"] = dict(collections.Counter(r["verdict_v2"] for r in rows))
    res["changed"] = [
        f"{r['graph']} {r['node']}: {r['verdict_v1']} -> {r['verdict_v2']}"
        for r in rows
        if r["verdict_v1"] != r["verdict_v2"]
    ]
    print(
        "SUMMARY",
        json.dumps({k: res[k] for k in ("counts_v1", "counts_v2")}),
        flush=True,
    )
json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
