"""Graph-edge audit runner, per exploratory/pilot-v1-spec.md.
usage: audit.py <out.json> [--baseline-only]"""

import glob, hashlib, json, os, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = "/mnt/d/AI/cache/huggingface/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"
TC = (
    "/mnt/ai/models/gemma-scope-transcoders/layer_{}/width_16k/average_l0_{}/params.npz"
)
L0 = {19: 12, 20: 11, 21: 13, 23: 25, 24: 37, 25: 41}
GRAPHS = [
    "gemma-fact-dallas-austin",
    "gemma-fact-oakland-sacramento",
    "gemma-small-big-en",
    "gemma-small-big-fr",
    "gemma-addition",
]
out_path = sys.argv[1]
BASE_ONLY = "--baseline-only" in sys.argv
dev = "cuda"
corpora = json.load(open(f"{HERE}/pilot-v1-corpora.json"))
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


st = {"L": None, "edit": None, "x": {}, "final": None, "precap": None}


def mk(L):
    def pre(m, args):
        r = args[0][0, -1].detach()
        st["x"][L] = r * torch.rsqrt(r.pow(2).mean() + 1e-6)

    def post(m, args, out):
        if st["L"] != L or st["edit"] is None:
            return out
        out = out.clone()
        out[0, -1] += st["edit"]
        return out

    lay = model.model.layers[L]
    lay.pre_feedforward_layernorm.register_forward_pre_hook(pre)
    lay.post_feedforward_layernorm.register_forward_hook(post)


for L in L0:
    mk(L)
model.model.norm.register_forward_pre_hook(
    lambda m, a: st.__setitem__("final", a[0][0, -1].detach())
)
model.lm_head.register_forward_hook(
    lambda m, a, o: st.__setitem__("precap", o[0, -1].detach())
)


@torch.no_grad()
def fwd(text, L=None, edit=None):
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    st["L"], st["edit"] = L, edit
    lg = model(ids).logits[0, -1]
    st["L"], st["edit"] = None, None
    return lg


def coeffs(L):
    t = tc(L)
    pre = st["x"][L] @ t["W_enc"] + t["b_enc"]
    return pre * (pre > t["threshold"])


def pre(text, L=None, edit=None):
    """Pre-softcap logits (the unit graph edge weights are in)."""
    fwd(text, L, edit)
    return st["precap"].clone()


def cen(lg, t):
    return (lg[t] - lg.mean()).item()


def rand_dir(seed, u, norm):
    rng = np.random.Generator(np.random.PCG64(seed))
    r = torch.tensor(rng.standard_normal(u.shape[0]), dtype=torch.float32, device=dev)
    r = r - (r @ u) * u
    return r / r.norm() * norm


def select_edges(g):
    N = {n["node_id"]: n for n in g["nodes"]}
    lg = sorted(
        [n for n in g["nodes"] if n["feature_type"] == "logit"],
        key=lambda n: -n["token_prob"],
    )
    tgt = lg[0]["node_id"]
    e = sorted(
        [
            (l["weight"], l["source"])
            for l in g["links"]
            if l["target"] == tgt
            and N[l["source"]]["feature_type"] == "cross layer transcoder"
        ],
        reverse=True,
    )[:3]
    return lg, [(w, N[s]) for w, s in e]


@torch.no_grad()
def test_edge(slug, g, lg_nodes, W, node):
    L, f, pos = (int(v) for v in node["node_id"].split("_"))
    text = g["metadata"]["prompt"].removeprefix("<bos>")
    ids = tok(text, return_tensors="pt").input_ids[0]
    assert tok.convert_ids_to_tokens(ids) == [
        t.replace(" ", "▁") for t in g["metadata"]["prompt_tokens"]
    ], (tok.convert_ids_to_tokens(ids), g["metadata"]["prompt_tokens"])
    assert pos == len(ids) - 1, "edge not at final position"
    tgt, run = (
        int(lg_nodes[0]["node_id"].split("_")[1]),
        int(lg_nodes[1]["node_id"].split("_")[1]),
    )
    t = tc(L)
    Wd = t["W_dec"]
    lg0 = fwd(text)
    c = coeffs(L).clone()
    final, precap = st["final"].clone(), st["precap"].clone()
    act = c[f].item()
    row = dict(
        graph=slug,
        node=node["node_id"],
        layer=L,
        feature=f,
        W=W,
        graph_activation=node["activation"],
        activation=act,
        target=tok.decode([tgt]),
        runner_up=tok.decode([run]),
        p_target=torch.softmax(lg0, -1)[tgt].item(),
    )
    row["identity_ok"] = bool(
        abs(act - node["activation"]) <= max(0.05, 0.01 * abs(node["activation"]))
    )
    if not row["identity_ok"]:
        row["verdict"] = "UNTESTABLE"
        return row
    # direct-path sanity: removed contribution through frozen final norm + unembed
    contrib = act * Wd[f]
    nw = model.model.norm.weight
    dl = (
        contrib * torch.rsqrt(final.pow(2).mean() + 1e-6) * (1 + nw)
    ) @ model.lm_head.weight.T
    row["direct_nocap"] = (dl[tgt] - dl.mean()).item()
    dlc = dl * (1 - torch.tanh(precap / 30) ** 2)
    row["direct_cap"] = (dlc[tgt] - dlc.mean()).item()
    if BASE_ONLY:
        return row
    base_c, base_g = cen(precap, tgt), (precap[tgt] - precap[run]).item()

    def eff(edit):
        lg = pre(text, L, edit)
        return base_c - cen(lg, tgt), base_g - (lg[tgt] - lg[run]).item()

    row["noop"] = (fwd(text, L, torch.zeros_like(contrib)) - lg0).abs().max().item()
    E, Eg = eff(-contrib)
    E_post = cen(lg0, tgt) - cen(fwd(text, L, -contrib), tgt)
    tn = contrib.norm().item()
    u = Wd[f] / Wd[f].norm()
    R = [eff(rand_dir(20261123 + i, u, tn))[0] for i in range(16)]
    norms = c * Wd.norm(dim=1)
    gap = (norms - tn).abs()
    gap[c <= 0] = float("inf")
    gap[f] = float("inf")
    gidx = int(gap.argmin())
    M1 = eff(-c[gidx] * Wd[gidx])[0]
    M2 = eff(-tn * Wd[gidx] / Wd[gidx].norm())[0]
    row.update(
        E=E,
        E_postcap=E_post,
        E_gap=Eg,
        ratio=E / W,
        C_R_mean=float(np.mean(R)),
        C_R_max=float(np.max(R)),
        matched_feature=gidx,
        matched_norm_ratio=norms[gidx].item() / tn,
        C_M1=M1,
        C_M2=M2,
        removed_norm=tn,
    )
    ctrl = E - row["C_R_mean"] >= 0.25 and E - M2 >= 0.25 and E > row["C_R_max"]
    if E <= 0:
        v = "WRONG_SIGN"
    elif not ctrl:
        v = "NOT_DISTINCT"
    elif E / W < 0.5:
        v = "WEAKER"
    elif E / W > 2:
        v = "STRONGER"
    else:
        v = "HOLDS"
    row["verdict"] = v
    # generality
    var = []
    for i, p in enumerate(corpora[slug]["variants"]):
        l0_ = fwd(p)
        pc0 = st["precap"].clone()
        cv = coeffs(L)[f].item()
        pt = torch.softmax(l0_, -1)[tgt].item()
        d = dict(prompt=p, p_target=pt, valid=bool(pt >= 0.1), coef=cv, E=0.0, C_R=0.0)
        if cv > 1e-6:
            con = cv * Wd[f]
            b = cen(pc0, tgt)
            d["E"] = b - cen(pre(p, L, -con), tgt)
            d["C_R"] = b - cen(
                pre(p, L, rand_dir(20261200 + i, u, con.norm().item())), tgt
            )
        var.append(d)
    vv = [d for d in var if d["valid"]]
    row["variants"] = var
    row["gen"] = dict(
        n_valid=len(vv),
        active_frac=sum(d["coef"] > 1e-6 for d in vv) / len(vv) if vv else None,
        median_E=float(np.median([d["E"] for d in vv])) if vv else None,
        beats_random_frac=sum(d["E"] > d["C_R"] for d in vv) / len(vv) if vv else None,
    )
    # specificity
    spec = []
    for kind, ps in (
        ("contrast", corpora[slug]["contrast"]),
        ("unrelated", corpora["_unrelated"]),
    ):
        for p in ps:
            l0_ = fwd(p)
            cv = coeffs(L)[f].item()
            top = torch.topk(l0_, 2).indices
            ch = 0.0
            if cv > 1e-6:
                la = fwd(p, L, -cv * Wd[f])
                ch = ((la[top[0]] - la[top[1]]) - (l0_[top[0]] - l0_[top[1]])).item()
            spec.append(
                dict(
                    kind=kind,
                    prompt=p,
                    coef=cv,
                    top=tok.decode([top[0]]),
                    gap_change=ch,
                )
            )
    row["specificity"] = spec
    row["spec"] = dict(
        active_frac=sum(s["coef"] > 1e-6 for s in spec) / len(spec),
        max_abs_gap_change=max(abs(s["gap_change"]) for s in spec),
    )
    row["spec"]["specific"] = bool(row["spec"]["max_abs_gap_change"] <= 0.1)
    return row


res = dict(graphs={}, edges=[])
for slug in GRAPHS:
    path = f"{HERE}/graphs/{slug}.json"
    res["graphs"][slug] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    g = json.load(open(path))
    lg_nodes, edges = select_edges(g)
    for W, node in edges:
        row = test_edge(slug, g, lg_nodes, W, node)
        res["edges"].append(row)
        print(
            json.dumps(
                {
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in row.items()
                    if k not in ("variants", "specificity")
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
