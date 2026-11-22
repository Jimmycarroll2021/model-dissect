"""EXPLORATORY (not the preregistered/admitted run): ablate Gemma-2-2b GemmaScope layer-20 transcoder
feature 15589 at the final token and measure the Austin-vs-Sacramento logit contrast.
Design follows specs/004-feature-attribution-validation/selected-claim-v1.md; fp32 GPU eager, no admission."""

import json, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SNAP = "/mnt/d/AI/cache/huggingface/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"
ROOT = "/mnt/ai/workspaces"
F, L = 15589, 20
corpus = json.load(
    open(
        f"{ROOT}/model-dissect-main/specs/004-feature-attribution-validation/corpus-v1.json"
    )
)
P = np.load(f"{ROOT}/model-dissect-google-l20-static-v1/params.npz")
dev = "cuda"
tok = AutoTokenizer.from_pretrained(SNAP)
model = (
    AutoModelForCausalLM.from_pretrained(
        SNAP, torch_dtype=torch.float32, attn_implementation="eager"
    )
    .to(dev)
    .eval()
)
W_enc = torch.tensor(P["W_enc"], device=dev)
W_dec = torch.tensor(P["W_dec"], device=dev)
b_enc = torch.tensor(P["b_enc"], device=dev)
thr = torch.tensor(P["threshold"], device=dev)
AUS = tok.encode(" Austin", add_special_tokens=False)
SAC = tok.encode(" Sacramento", add_special_tokens=False)
print(
    "tokens Austin",
    AUS,
    "Sacramento",
    SAC,
    "graph id",
    corpus["target_token_id_graph"],
    flush=True,
)
AUS, SAC = AUS[0], SAC[0]

state = {"edit": None, "x": None}
mlp = model.model.layers[L].mlp


def pre_hook(m, args):
    state["x"] = args[0][
        0, -1
    ].detach()  # MLP input (post pre_feedforward_layernorm), final position


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
def contrast(text, edit=None):
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    state["edit"] = edit
    lg = model(ids).logits[0, -1]
    state["edit"] = None
    return (lg[AUS] - lg[SAC]).item(), lg


@torch.no_grad()
def coef_at(text):
    contrast(text)
    return coeffs(state["x"])


def ablate(f):
    return lambda x: -coeffs(x)[f] * W_dec[f]


@torch.no_grad()
def run():
    res = {"pairs": [], "noop_max_diff": 0.0}
    tx_all = []
    for i, p in enumerate(corpus["pairs"]):
        tx, ca = p["texas"], p["california"]
        c_tx = coef_at(tx)
        c_ca = coef_at(ca)
        d0, lg0 = contrast(tx)
        dca0, _ = contrast(ca)
        # no-op / identity: zero edit must reproduce baseline exactly
        dn, lgn = contrast(tx, lambda x: torch.zeros_like(W_dec[0]))
        res["noop_max_diff"] = max(res["noop_max_diff"], (lgn - lg0).abs().max().item())
        ct = c_tx[F].item()
        tgt_norm = (ct * W_dec[F]).norm().item()
        dA, _ = contrast(tx, ablate(F))
        # matched feature: smallest ID != F, coefficient and decoded norm within 20% of target on Texas baseline
        mag = c_tx.abs()
        dn_ = mag * W_dec.norm(dim=1)
        ok = (
            (mag > 0)
            & ((mag - abs(ct)).abs() <= 0.2 * abs(ct))
            & ((dn_ - tgt_norm).abs() <= 0.2 * tgt_norm)
        )
        ok[F] = False
        cand = torch.nonzero(ok).flatten()
        mf = int(cand[0]) if len(cand) else None
        dM = contrast(tx, ablate(mf))[0] if mf is not None else None
        # random residual edit: PCG64 seed 20261119+i, orthogonal to target decoder dir, norm = target ablation norm
        rng = np.random.Generator(np.random.PCG64(20261119 + i))
        r = torch.tensor(
            rng.standard_normal(W_dec.shape[1]), dtype=torch.float32, device=dev
        )
        u = W_dec[F] / W_dec[F].norm()
        r = r - (r @ u) * u
        r = r / r.norm() * tgt_norm
        dR = contrast(tx, lambda x: r)[0]
        # California specificity + cross patch (secondary)
        dca_abl = contrast(ca, ablate(F))[0]
        cca = c_ca[F].item()
        cross = contrast(tx, lambda x: (cca - coeffs(x)[F]) * W_dec[F])[0]
        row = dict(
            id=p["id"],
            coef_tx=ct,
            coef_ca=cca,
            D_tx=d0,
            D_ca=dca0,
            D_tx_abl=dA,
            D_tx_matched=dM,
            matched_feature=mf,
            D_tx_random=dR,
            D_ca_abl=dca_abl,
            D_tx_cross=cross,
            target_norm=tgt_norm,
        )
        row.update(
            A=d0 - dA, C_matched=None if dM is None else d0 - dM, C_random=d0 - dR
        )
        row["Q_matched"] = None if dM is None else row["A"] - row["C_matched"]
        row["Q_random"] = row["A"] - row["C_random"]
        res["pairs"].append(row)
        print(
            json.dumps(
                {
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in row.items()
                }
            ),
            flush=True,
        )
    # non-target specificity
    nt = []
    for n in corpus["non_target"]:
        a = tok.encode(n["answer"], add_special_tokens=False)[0]
        d = tok.encode(n["distractor"], add_special_tokens=False)[0]

        def cm(edit):
            ids = tok(n["prompt"], return_tensors="pt").input_ids.to(dev)
            state["edit"] = edit
            lg = model(ids).logits[0, -1]
            state["edit"] = None
            return (lg[a] - lg[d]).item()

        b = cm(None)
        e = cm(ablate(F))
        nt.append(
            dict(
                id=n["id"],
                base=b,
                ablated=e,
                delta=e - b,
                coef=coef_at(n["prompt"])[F].item(),
            )
        )
    res["non_target"] = nt
    # discovery prompt
    dd = corpus["discovery_excluded"]
    res["dallas"] = dict(
        coef=coef_at(dd)[F].item(), D=contrast(dd)[0], D_abl=contrast(dd, ablate(F))[0]
    )
    return res


res = run()
rows = res["pairs"]
elig_tx = [r for r in rows if r["coef_tx"] > 1e-6 and r["D_tx"] >= 0.5]
elig_ca = [r for r in rows if -r["D_ca"] >= 0.5]
A = np.array([r["A"] for r in rows])
summ = dict(
    n_pairs=len(rows),
    eligible_texas=len(elig_tx),
    eligible_california=len(elig_ca),
    median_A=float(np.median(A)),
    median_Q_matched=float(
        np.median([r["Q_matched"] for r in rows if r["Q_matched"] is not None])
    )
    if any(r["Q_matched"] is not None for r in rows)
    else None,
    count_Q_matched_pos=sum(
        1 for r in rows if r["Q_matched"] is not None and r["Q_matched"] > 0
    ),
    matched_found=sum(1 for r in rows if r["matched_feature"] is not None),
    median_Q_random=float(np.median([r["Q_random"] for r in rows])),
    count_Q_random_pos=sum(1 for r in rows if r["Q_random"] > 0),
    max_abs_ca_change=max(abs(r["D_ca_abl"] - r["D_ca"]) for r in rows),
    max_abs_nontarget_change=max(abs(n["delta"]) for n in res["non_target"]),
    noop_max_logit_diff=res["noop_max_diff"],
)
res["summary"] = summ
json.dump(
    res,
    open(f"{ROOT}/model-dissect-main/exploratory/ablation-results.json", "w"),
    indent=1,
)
print("SUMMARY", json.dumps(summ, indent=1))
