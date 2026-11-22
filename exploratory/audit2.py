"""Graph-edge audit runner, per exploratory/scale-v1-spec.md (any position, all layers, no variants).
usage: audit2.py <out.json> [--baseline-only]"""

import glob, hashlib, json, os, sys, numpy as np, torch
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
name = lambda p: os.path.basename(p)[:-5]
NEW = sorted(glob.glob(f"{HERE}/graphs2/*.json"))
PILOT = sorted(glob.glob(f"{HERE}/graphs/*.json"))
UNRELATED = json.load(open(f"{HERE}/pilot-v1-corpora.json"))["_unrelated"]
out_path = sys.argv[1]
BASE_ONLY = "--baseline-only" in sys.argv
dev = "cuda"
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


st = {"L": None, "edit": None, "pos": -1, "x": {}, "final": None, "precap": None}


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
def fwd(text, L=None, edit=None, pos=-1):
    """Real (capped) final-position logits; st['precap'] holds the pre-cap ones."""
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    st["L"], st["edit"], st["pos"] = L, edit, pos
    lg = model(ids).logits[0, -1]
    st["L"], st["edit"] = None, None
    return lg


def pre(text, L=None, edit=None, pos=-1):
    fwd(text, L, edit, pos)
    return st["precap"].clone()


def coeffs(L):
    t = tc(L)
    p = st["x"][L] @ t["W_enc"] + t["b_enc"]
    return p * (p > t["threshold"])


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


def text_of(g):
    return g["metadata"]["prompt"].removeprefix("<bos>")


def tokens_ok(g):
    ids = tok(text_of(g), return_tensors="pt").input_ids[0]
    return [tok.decode([i]) for i in ids] == list(g["metadata"]["prompt_tokens"]), len(
        ids
    )


@torch.no_grad()
def test_edge(slug, g, lg_nodes, W, node, n_tok, others):
    L, f, pos = (int(v) for v in node["node_id"].split("_"))
    ga = node.get("activation")  # absent in some older graphs: identity gate then fails
    text = text_of(g)
    at_final = pos == n_tok - 1
    p = pos - n_tok  # negative index
    tgt, run = (int(n["node_id"].split("_")[1]) for n in lg_nodes[:2])
    Wd = tc(L)["W_dec"]
    lg0 = fwd(text, pos=p)
    c = coeffs(L).clone()
    final, precap = st["final"].clone(), st["precap"].clone()
    act = c[f].item()
    row = dict(
        graph=slug,
        node=node["node_id"],
        layer=L,
        feature=f,
        at_final=at_final,
        W=W,
        graph_activation=ga,
        activation=act,
        target=tok.decode([tgt]),
        runner_up=tok.decode([run]),
        p_target=torch.softmax(lg0, -1)[tgt].item(),
    )
    row["identity_ok"] = bool(ga is not None and abs(act - ga) <= max(0.05, 0.01 * abs(ga)))
    if not row["identity_ok"]:
        row["verdict"] = "UNTESTABLE"
        return row
    contrib = act * Wd[f]
    if at_final:
        dl = (
            contrib
            * torch.rsqrt(final.pow(2).mean() + 1e-6)
            * (1 + model.model.norm.weight)
        ) @ model.lm_head.weight.T
        row["direct_nocap"] = (dl[tgt] - dl.mean()).item()
    if BASE_ONLY:
        return row
    base_c, base_g = cen(precap, tgt), (precap[tgt] - precap[run]).item()

    def eff(edit):
        lg = pre(text, L, edit, p)
        return base_c - cen(lg, tgt), base_g - (lg[tgt] - lg[run]).item()

    row["noop"] = (fwd(text, L, torch.zeros_like(contrib), p) - lg0).abs().max().item()
    E, Eg = eff(-contrib)
    E_post = cen(lg0, tgt) - cen(fwd(text, L, -contrib, p), tgt)
    tn = contrib.norm().item()
    u = Wd[f] / Wd[f].norm()
    R = [eff(rand_dir(20261123 + i, u, tn))[0] for i in range(16)]
    norms = c * Wd.norm(dim=1)
    gap = (norms - tn).abs()
    gap[c <= 0] = float("inf")
    gap[f] = float("inf")
    gi = int(gap.argmin())
    has_m = bool(torch.isfinite(gap[gi]))
    M1 = eff(-c[gi] * Wd[gi])[0] if has_m else None
    M2 = eff(-tn * Wd[gi] / Wd[gi].norm())[0] if has_m else None
    row.update(
        E=E,
        E_postcap=E_post,
        E_gap=Eg,
        ratio=E / W,
        C_R_mean=float(np.mean(R)),
        C_R_max=float(np.max(R)),
        matched_feature=gi if has_m else None,
        matched_norm_ratio=norms[gi].item() / tn if has_m else None,
        C_M1=M1,
        C_M2=M2,
        removed_norm=tn,
    )
    ctrl = (
        E - row["C_R_mean"] >= 0.25 and has_m and E - M2 >= 0.25 and E > row["C_R_max"]
    )
    row["verdict"] = (
        "WRONG_SIGN"
        if E <= 0
        else "NOT_DISTINCT"
        if not ctrl
        else "WEAKER"
        if E / W < 0.5
        else "STRONGER"
        if E / W > 2
        else "HOLDS"
    )
    if at_final:
        spec = []
        for q in others:
            l0_ = fwd(q)
            cv = coeffs(L)[f].item()
            top = torch.topk(l0_, 2).indices
            ch = 0.0
            if cv > 1e-6:
                la = fwd(q, L, -cv * Wd[f])
                ch = ((la[top[0]] - la[top[1]]) - (l0_[top[0]] - l0_[top[1]])).item()
            spec.append(dict(prompt=q, coef=cv, gap_change=ch))
        row["specificity"] = spec
        m = max(abs(s["gap_change"]) for s in spec)
        row["spec"] = dict(
            active_frac=sum(s["coef"] > 1e-6 for s in spec) / len(spec),
            max_abs_gap_change=m,
            specific=bool(m <= 0.1),
        )
    return row


ALL = {name(p): json.load(open(p)) for p in PILOT + NEW}
res = dict(graphs={}, edges=[])
for path in NEW:
    slug = name(path)
    g = ALL[slug]
    res["graphs"][slug] = dict(
        sha256=hashlib.sha256(open(path, "rb").read()).hexdigest()
    )
    ok, n_tok = tokens_ok(g)
    if not ok:
        res["graphs"][slug]["status"] = "UNTESTABLE_TOKENISATION"
        print(slug, "UNTESTABLE_TOKENISATION", flush=True)
        continue
    others = [text_of(h) for s, h in ALL.items() if s != slug] + UNRELATED
    lg_nodes, edges = select_edges(g)
    rows = [test_edge(slug, g, lg_nodes, W, node, n_tok, others) for W, node in edges]
    fin = [r for r in rows if r.get("direct_nocap") is not None]
    unit = (
        max(fin, key=lambda r: r["layer"])["direct_nocap"]
        / max(fin, key=lambda r: r["layer"])["W"]
        if fin
        else None
    )
    res["graphs"][slug]["unit_ratio"] = unit
    bad = unit is not None and not (0.5 <= unit <= 2)
    res["graphs"][slug]["status"] = "UNIT_MISMATCH" if bad else "TESTED"
    for r in rows:
        if bad:
            r["verdict_before_unit_gate"] = r.get("verdict")
            r["verdict"] = "UNIT_MISMATCH"
        res["edges"].append(r)
        print(
            json.dumps(
                {
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in r.items()
                    if k != "specificity"
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
