"""Paraphrase robustness runner, per exploratory/paraphrase-v1-spec.md.
usage: paraphrase.py <out.json> [--baseline-only]
       paraphrase.py --table <run.json> <out.md>      (no model; writes the per-edge table)
Model, hooks, convention, controls and edge selection are copied from audit2.py
(which runs on import, so it is not imported)."""

import glob, json, os, sys, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = {"dallas-austin", "gemma-it-pirate"}  # untestable in scale v1
DISTINCT_OWN = {"HOLDS", "WEAKER", "STRONGER"}
OUTCOMES = ["DISTINCT", "NOT_DISTINCT", "WRONG_SIGN", "INACTIVE"]


def write_table(run_path, md_path):
    R = json.load(open(run_path))
    f2 = lambda v: "" if v is None else f"{v:.2f}"
    L = [
        "# Paraphrase v1: per-edge results (5 paraphrases per graph, final-position read and removal)",
        "",
        f"Edges tested: {R['overall']['n_edges_tested']}; position not transferable: "
        f"{R['overall']['n_position_not_transferable']}. Agreement of own-prompt category with "
        f"paraphrase majority: {R['overall']['agree']}/{R['overall']['n_edges_compared']} "
        f"= {R['overall']['agreement']:.2f}. Per-paraphrase outcomes over valid pairs: "
        + ", ".join(f"{k} {v}" for k, v in R["overall"]["outcome_counts"].items())
        + ".",
        "",
        "| graph | edge | own verdict | own E | valid | active | distinct | median E | median drop in p | D / ND / WS / IN | majority | agrees |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in R["edges"]:
        lf = f"{e['layer']}/{e['feature']}"
        if e["status"] != "TESTED":
            L.append(
                f"| {e['graph']} | {lf} (earlier token) | {e['own_verdict']} | {f2(e['own_E'])} "
                f"| | | | | | | position not transferable | |"
            )
            continue
        s = e["summary"]
        oc = " / ".join(str(s["outcome_counts"][k]) for k in OUTCOMES)
        L.append(
            f"| {e['graph']} | {lf} | {e['own_verdict']} | {f2(e['own_E'])} | {s['n_valid']} "
            f"| {f2(s['active_frac'])} | {f2(s['distinct_frac'])} | {f2(s['median_E'])} "
            f"| {f2(s['median_dp'])} | {oc} | {s['majority'] or 'n/a'}{' (tie)' if s['tie'] else ''} "
            f"| {'' if s['agrees'] is None else ('yes' if s['agrees'] else 'NO')} |"
        )
    L += [
        "",
        "D = DISTINCT, ND = NOT_DISTINCT, WS = WRONG_SIGN, IN = INACTIVE (counts over valid paraphrases). "
        "Median E and median drop in p count inactive paraphrases as 0. Own E is reference only.",
        "",
    ]
    open(md_path, "w").write("\n".join(L))


if sys.argv[1] == "--table":
    write_table(sys.argv[2], sys.argv[3])
    sys.exit(0)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
PATHS = sorted(glob.glob(f"{HERE}/graphs/*.json")) + sorted(
    glob.glob(f"{HERE}/graphs2/*.json")
)
GRAPHS = {name(p): json.load(open(p)) for p in PATHS if name(p) not in SKIP}
CORP = json.load(open(f"{HERE}/paraphrase-v1-corpora.json"))
assert set(CORP) == set(GRAPHS) and all(len(v) == 5 for v in CORP.values())
OWN = {
    (r["graph"], r["node"]): r
    for f in ("pilot-v1-run1.json", "scale-v1-run1.json")
    for r in json.load(open(f"{HERE}/results/{f}"))["edges"]
}
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


def n_tokens(text):
    return len(tok(text).input_ids)


@torch.no_grad()
def baseline(text, tgt):
    lg0 = fwd(text)
    return dict(
        prompt=text,
        n_tok=n_tokens(text),
        p_target=torch.softmax(lg0, -1)[tgt].item(),
        top1=tok.decode([int(lg0.argmax())]),
    )


@torch.no_grad()
def measure(text, L, f, tgt, pos=-1):
    """Read feature f of layer L at position pos, remove its decoded contribution there,
    and measure the final-position effect on target tgt, with the audit2 controls."""
    Wd = tc(L)["W_dec"]
    lg0 = fwd(text, pos=pos)
    c = coeffs(L).clone()
    precap = st["precap"].clone()
    p0 = torch.softmax(lg0, -1)[tgt].item()
    act = c[f].item()
    d = dict(coef=act)
    if act <= 1e-6:
        return d
    contrib = act * Wd[f]
    base_c = cen(precap, tgt)

    def eff(edit):
        return base_c - cen(pre(text, L, edit, pos), tgt)

    E = eff(-contrib)
    p1 = torch.softmax(fwd(text, L, -contrib, pos), -1)[tgt].item()
    tn = contrib.norm().item()
    u = Wd[f] / Wd[f].norm()
    R = [eff(rand_dir(20261123 + i, u, tn)) for i in range(16)]
    norms = c * Wd.norm(dim=1)
    gap = (norms - tn).abs()
    gap[c <= 0] = float("inf")
    gap[f] = float("inf")
    gi = int(gap.argmin())
    has_m = bool(torch.isfinite(gap[gi]))
    M2 = eff(-tn * Wd[gi] / Wd[gi].norm()) if has_m else None
    d.update(
        E=E,
        dp=p0 - p1,
        C_R_mean=float(np.mean(R)),
        C_R_max=float(np.max(R)),
        matched_feature=gi if has_m else None,
        C_M2=M2,
    )
    return d


def outcome(d):
    if d["coef"] <= 1e-6:
        return "INACTIVE"
    E = d["E"]
    if E <= 0:
        return "WRONG_SIGN"
    ok = (
        E - d["C_R_mean"] >= 0.25
        and d["C_M2"] is not None
        and E - d["C_M2"] >= 0.25
        and E > d["C_R_max"]
    )
    return "DISTINCT" if ok else "NOT_DISTINCT"


res = dict(baseline={}, reproduction=None, edges=[], overall=None)
tgt_of = {}
for slug, g in GRAPHS.items():
    lg_nodes, _ = select_edges(g)
    tgt = int(lg_nodes[0]["node_id"].split("_")[1])
    tgt_of[slug] = tgt
    rows = [baseline(t, tgt) for t in CORP[slug]]
    for r in rows:
        r["valid"] = bool(r["p_target"] >= 0.1)
    res["baseline"][slug] = dict(
        target=tok.decode([tgt]),
        n_valid=sum(r["valid"] for r in rows),
        paraphrases=rows,
    )
    print(
        slug,
        repr(tok.decode([tgt])),
        res["baseline"][slug]["n_valid"],
        [(round(r["p_target"], 3), r["top1"]) for r in rows],
        flush=True,
    )

if BASE_ONLY:
    json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
    sys.exit(0)

# Reproduction gate: own-prompt E for every edge must match the stored result within 1e-3.
rep = []
for slug, g in GRAPHS.items():
    _, edges = select_edges(g)
    text = text_of(g)
    n = n_tokens(text)
    for W, node in edges:
        L, f, pos = (int(v) for v in node["node_id"].split("_"))
        d = measure(text, L, f, tgt_of[slug], pos - n)
        stored = OWN[(slug, node["node_id"])]["E"]
        rep.append(
            dict(
                graph=slug,
                node=node["node_id"],
                E=d["E"],
                stored_E=stored,
                diff=abs(d["E"] - stored),
            )
        )
mx = max(r["diff"] for r in rep)
res["reproduction"] = dict(n=len(rep), max_abs_diff=mx, ok=bool(mx <= 1e-3), rows=rep)
print("reproduction", len(rep), "max |dE| =", mx, flush=True)
if mx > 1e-3:
    json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
    sys.exit("reproduction gate FAILED; no paraphrase interventions run")

counts = {k: 0 for k in OUTCOMES}
agree = compared = 0
for slug, g in GRAPHS.items():
    _, edges = select_edges(g)
    n_own = n_tokens(text_of(g))
    for W, node in edges:
        L, f, pos = (int(v) for v in node["node_id"].split("_"))
        own = OWN[(slug, node["node_id"])]
        row = dict(
            graph=slug,
            node=node["node_id"],
            layer=L,
            feature=f,
            W=W,
            own_verdict=own["verdict"],
            own_E=own["E"],
        )
        if pos != n_own - 1:
            row["status"] = "POSITION_NOT_TRANSFERABLE"
            res["edges"].append(row)
            print(json.dumps(row), flush=True)
            continue
        row["status"] = "TESTED"
        P = []
        for b in res["baseline"][slug]["paraphrases"]:
            if not b["valid"]:
                P.append(dict(prompt=b["prompt"], outcome="INVALID"))
                continue
            d = measure(b["prompt"], L, f, tgt_of[slug])
            d = dict(prompt=b["prompt"], **d)
            d["outcome"] = outcome(d)
            P.append(d)
        row["paraphrases"] = P
        V = [d for d in P if d["outcome"] != "INVALID"]
        oc = {k: sum(d["outcome"] == k for d in V) for k in OUTCOMES}
        for k in OUTCOMES:
            counts[k] += oc[k]
        nv = len(V)
        maj = (
            None
            if nv == 0
            else ("distinct" if oc["DISTINCT"] > nv / 2 else "not distinct")
        )
        own_cat = "distinct" if own["verdict"] in DISTINCT_OWN else "not distinct"
        ag = None if maj is None else maj == own_cat
        if ag is not None:
            compared += 1
            agree += ag
        row["summary"] = dict(
            n_valid=nv,
            active_frac=sum(d["coef"] > 1e-6 for d in V) / nv if nv else None,
            distinct_frac=oc["DISTINCT"] / nv if nv else None,
            median_E=float(np.median([d.get("E", 0.0) for d in V])) if nv else None,
            median_dp=float(np.median([d.get("dp", 0.0) for d in V])) if nv else None,
            outcome_counts=oc,
            own_category=own_cat,
            majority=maj,
            tie=bool(nv > 0 and oc["DISTINCT"] * 2 == nv),
            agrees=ag,
        )
        res["edges"].append(row)
        print(
            slug,
            node["node_id"],
            own["verdict"],
            json.dumps(row["summary"]),
            flush=True,
        )

tested = [e for e in res["edges"] if e["status"] == "TESTED"]
res["overall"] = dict(
    n_edges=len(res["edges"]),
    n_edges_tested=len(tested),
    n_position_not_transferable=len(res["edges"]) - len(tested),
    n_edges_compared=compared,
    agree=agree,
    agreement=agree / compared if compared else None,
    outcome_counts=counts,
    disagreeing=[
        f"{e['graph']} {e['node']}" for e in tested if e["summary"]["agrees"] is False
    ],
)
print(json.dumps(res["overall"]), flush=True)
json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
