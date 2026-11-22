"""Feature-ablation run per exploratory/held-out-v2-spec.md.
usage: ablation_v2.py <corpus.json> <cuda|cpu> <out.json>"""

import json, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SNAP = "/mnt/d/AI/cache/huggingface/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"
PARAMS = "/mnt/ai/workspaces/model-dissect-google-l20-static-v1/params.npz"
F, L = 15589, 20
corpus_path, dev, out_path = sys.argv[1:4]
corpus = json.load(open(corpus_path))
P = np.load(PARAMS)
tok = AutoTokenizer.from_pretrained(SNAP)
model = (
    AutoModelForCausalLM.from_pretrained(
        SNAP, torch_dtype=torch.float32, attn_implementation="eager"
    )
    .to(dev)
    .eval()
)
W_enc, W_dec, b_enc, thr = (
    torch.tensor(P[k], device=dev) for k in ("W_enc", "W_dec", "b_enc", "threshold")
)


def single(text):
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == 1, (text, ids)
    return ids[0]


AUS, SAC = single(corpus["target_text"]), single(corpus["comparator_text"])
assert AUS == corpus["target_token_id_graph"]

state = {"edit": None, "x": None}
mlp = model.model.layers[L].mlp


def pre_hook(m, args):
    state["x"] = args[0][0, -1].detach()


def post_hook(m, args, out):
    if state["edit"] is None:
        return out
    out = out.clone()
    out[0, -1] += state["edit"](state["x"])
    return out


mlp.register_forward_pre_hook(pre_hook)
mlp.register_forward_hook(post_hook)


def coeffs(x):
    pre = x @ W_enc + b_enc
    return pre * (pre > thr)


@torch.no_grad()
def logits(text, edit=None):
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    state["edit"] = edit
    lg = model(ids).logits[0, -1]
    state["edit"] = None
    return lg


def D(text, edit=None):
    lg = logits(text, edit)
    return (lg[AUS] - lg[SAC]).item()


def ablate(f):
    return lambda x: -coeffs(x)[f] * W_dec[f]


@torch.no_grad()
def run():
    rows, noop = [], 0.0
    for i, p in enumerate(corpus["pairs"]):
        tx, ca = p["texas"], p["california"]
        lg0 = logits(tx)
        c_tx = coeffs(state["x"]).clone()
        d0 = (lg0[AUS] - lg0[SAC]).item()
        dca0 = D(ca)
        c_ca = coeffs(state["x"])[F].item()
        noop = max(
            noop,
            (logits(tx, lambda x: torch.zeros_like(W_dec[0])) - lg0).abs().max().item(),
        )
        ct = c_tx[F].item()
        row = dict(
            id=p["id"],
            coef_tx=ct,
            coef_ca=c_ca,
            D_tx=d0,
            D_ca=dca0,
            valid=bool(d0 >= 0.5 and -dca0 >= 0.5),
            active=bool(ct > 1e-6),
        )
        if row["active"]:
            tn = (ct * W_dec[F]).norm().item()
            dA = D(tx, ablate(F))
            rng = np.random.Generator(np.random.PCG64(20261119 + i))
            r = torch.tensor(
                rng.standard_normal(W_dec.shape[1]), dtype=torch.float32, device=dev
            )
            u = W_dec[F] / W_dec[F].norm()
            r = r - (r @ u) * u
            r = r / r.norm() * tn
            dR = D(tx, lambda x: r)
            norms = c_tx * W_dec.norm(dim=1)
            gap = (norms - tn).abs()
            gap[c_tx <= 0] = float("inf")
            gap[F] = float("inf")
            g = int(gap.argmin())
            gdir = W_dec[g] / W_dec[g].norm()
            dM1 = D(tx, ablate(g))
            dM2 = D(tx, lambda x: -tn * gdir)
            A = d0 - dA
            row.update(
                target_norm=tn,
                matched_feature=g,
                matched_norm_ratio=norms[g].item() / tn,
                A=A,
                Q_R=A - (d0 - dR),
                Q_M1=A - (d0 - dM1),
                Q_M2=A - (d0 - dM2),
                D_tx_abl=dA,
            )
        else:
            row.update(A=0.0, Q_R=0.0, Q_M1=0.0, Q_M2=0.0)
        row["ca_change"] = D(ca, ablate(F)) - dca0
        rows.append(row)
        print(
            json.dumps(
                {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
            ),
            flush=True,
        )
    nt = []
    for n in corpus["non_target"]:
        a, d = single(n["answer"]), single(n["distractor"])
        b = logits(n["prompt"])
        e = logits(n["prompt"], ablate(F))
        nt.append(dict(id=n["id"], delta=((e[a] - e[d]) - (b[a] - b[d])).item()))
    return rows, nt, noop


rows, nt, noop = run()
v = [r for r in rows if r["valid"]]
med = lambda k: float(np.median([r[k] for r in v])) if v else None
frac = lambda k: sum(r[k] > 0 for r in v) / len(v) if v else None
s = dict(
    n_pairs=len(rows),
    n_valid=len(v),
    invalid=[r["id"] for r in rows if not r["valid"]],
    activity_rate=sum(r["active"] for r in v) / len(v) if v else None,
    inactive=[r["id"] for r in v if not r["active"]],
    median_A=med("A"),
    controls={
        k: dict(median_Q=med("Q_" + k), frac_Q_pos=frac("Q_" + k))
        for k in ("R", "M1", "M2")
    },
    max_abs_ca_change=max(abs(r["ca_change"]) for r in v) if v else None,
    max_abs_nontarget_change=max(abs(n["delta"]) for n in nt),
    noop_max_logit_diff=noop,
)
if len(v) < 16:
    s["verdict"] = "INCOMPLETE"
else:
    ok = (
        s["median_A"] >= 0.5
        and all(
            c["median_Q"] >= 0.25 and c["frac_Q_pos"] >= 0.75
            for c in s["controls"].values()
        )
        and s["max_abs_ca_change"] <= 0.1
        and s["max_abs_nontarget_change"] <= 0.1
    )
    s["verdict"] = "PASS" if ok and noop == 0 else ("FAIL" if not ok else "INCOMPLETE")
json.dump(
    dict(corpus=corpus_path, device=dev, pairs=rows, non_target=nt, summary=s),
    open(out_path, "w"),
    indent=1,
)
print("SUMMARY", json.dumps(s, indent=1))
