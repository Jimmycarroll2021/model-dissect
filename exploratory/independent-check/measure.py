"""Independent re-measurement of graph-edge ablation effects in Gemma-2-2b.
Written from pilot-v1-spec.md / scale-v1-spec.md only."""

import json, sys, glob, os
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP = "/mnt/ai/workspaces/model-dissect-main/exploratory"
SNAP = "/mnt/d/AI/cache/huggingface/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"
TC = "/mnt/ai/models/gemma-scope-transcoders"
OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(20261123, 20261139))
EDGES = {
    "graphs/gemma-fact-dallas-austin": ["20_15589_10", "21_5943_10"],
    "graphs/gemma-addition": ["24_14176_6", "25_14682_6"],
    "graphs/gemma-small-big-en": ["21_10062_8"],
    "graphs2/gemma-bee-insect": ["22_16038_6", "14_11504_2", "19_5460_6"],
    "graphs2/gemma-dollar": ["18_10102_6", "20_5796_6"],
    "graphs2/gemma-girls-are": ["22_11517_6"],
    "graphs2/gemma-small-big-zh": ["24_2394_8", "23_8488_8"],
    "graphs2/gemma-michael-jordan": ["21_9324_8"],
    "graphs2/gemma-fact-shanghai-beijing": ["19_12274_9"],
}

dev = "cuda"
tok = AutoTokenizer.from_pretrained(SNAP)
model = (
    AutoModelForCausalLM.from_pretrained(
        SNAP, torch_dtype=torch.float32, attn_implementation="eager"
    )
    .to(dev)
    .eval()
)
cfg = model.config
assert cfg.final_logit_softcapping == 30.0, cfg.final_logit_softcapping
layers = model.model.layers
eps = cfg.rms_norm_eps
WU = model.lm_head.weight  # [V, d]

_tc = {}


def transcoder(L):
    if L not in _tc:
        (p,) = glob.glob(f"{TC}/layer_{L}/width_16k/average_l0_*/params.npz")
        z = np.load(p)
        _tc[L] = {
            k: torch.tensor(z[k], dtype=torch.float32, device=dev)
            for k in ["W_enc", "W_dec", "b_enc", "b_dec", "threshold"]
        }
        _tc[L]["path"] = p
    return _tc[L]


def rmsnorm_nogain(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def run(ids, edit=None):
    """edit = (layer, pos, vector) subtracted from post_feedforward_layernorm output.
    Returns precap logits at last pos, capped logits at last pos, captured pre-MLP inputs, final resid."""
    cap = {"premlp": {}, "final_in": None, "final_out": None}
    hs = []
    for i, ly in enumerate(layers):

        def pre_hook(mod, args, i=i):
            cap["premlp"][i] = args[0].detach()

        hs.append(ly.pre_feedforward_layernorm.register_forward_pre_hook(pre_hook))
    if edit is not None:
        L, pos, vec = edit

        def post_hook(mod, args, out):
            out = out.clone()
            out[0, pos] -= vec
            return out

        hs.append(layers[L].post_feedforward_layernorm.register_forward_hook(post_hook))

    def fn_hook(mod, args, out):
        cap["final_in"] = args[0].detach()
        cap["final_out"] = out.detach()

    hs.append(model.model.norm.register_forward_hook(fn_hook))
    try:
        with torch.no_grad():
            o = model(ids)
    finally:
        for h in hs:
            h.remove()
    pre = cap["final_out"][0, -1] @ WU.T
    post = o.logits[0, -1]
    # self-check: our precap reproduces the capped output
    assert torch.allclose(30 * torch.tanh(pre / 30), post, atol=1e-3), (
        (30 * torch.tanh(pre / 30) - post).abs().max()
    )
    return pre, post, cap


def mc(lg, t):
    return (lg[t] - lg.mean()).item()


rows = []
for gpath, nids in EDGES.items():
    g = json.load(open(f"{EXP}/{gpath}.json"))
    prompt = g["metadata"]["prompt"]
    assert prompt.startswith("<bos>")
    ids = tok(prompt[len("<bos>") :], return_tensors="pt").input_ids.to(dev)
    toks = [tok.decode([i]) for i in ids[0].tolist()]
    tok_ok = (
        toks == g["metadata"]["prompt_tokens"]
        or tok.convert_ids_to_tokens(ids[0]) == g["metadata"]["prompt_tokens"]
    )
    nodes = {n["node_id"]: n for n in g["nodes"]}
    logits_nodes = sorted(
        [n for n in g["nodes"] if n["feature_type"] == "logit"],
        key=lambda n: -n["token_prob"],
    )
    tgt_node = logits_nodes[0]
    t = int(tgt_node["node_id"].split("_")[1])
    last = ids.shape[1] - 1
    assert int(tgt_node["node_id"].split("_")[2]) == last
    pre0, post0, cap0 = run(ids)
    p0 = torch.softmax(post0, -1)
    base = dict(
        mc_pre=mc(pre0, t),
        mc_post=mc(post0, t),
        prob=p0[t].item(),
        graph_prob=tgt_node["token_prob"],
        top1=int(post0.argmax()),
    )
    W_of = {}
    for l in g["links"]:
        if l["target"] == tgt_node["node_id"]:
            W_of[l["source"]] = l["weight"]
    for nid in nids:
        L, f, pos = map(int, nid.split("_"))
        node = nodes[nid]
        T = transcoder(L)
        x = cap0["premlp"][L][0, pos]

        def enc(inp):
            pre = inp @ T["W_enc"][:, f] + T["b_enc"][f]
            return (pre * (pre > T["threshold"][f])).item()

        a_nogain = enc(rmsnorm_nogain(x))
        a_gain = enc(layers[L].pre_feedforward_layernorm(x))
        a_raw = enc(x)
        ga = node["activation"]
        ident_ok = ga is not None and (
            abs(a_nogain - ga) <= 0.05 or abs(a_nogain - ga) <= 0.01 * abs(ga)
        )
        contrib = a_nogain * T["W_dec"][f]
        pre1, post1, _ = run(ids, (L, pos, contrib))
        p1 = torch.softmax(post1, -1)
        E = base["mc_pre"] - mc(pre1, t)
        E_post = base["mc_post"] - mc(post1, t)
        dprob = p1[t].item() - base["prob"]
        # random-direction controls
        d = T["W_dec"][f] / T["W_dec"][f].norm()
        nrm = contrib.norm()
        rE = []
        for s in SEEDS:
            v = torch.tensor(
                np.random.Generator(np.random.PCG64(s)).standard_normal(
                    cfg.hidden_size
                ),
                dtype=torch.float32,
                device=dev,
            )
            v = v - (v @ d) * d
            v = v / v.norm() * nrm
            pr, _, _ = run(ids, (L, pos, v))
            rE.append(base["mc_pre"] - mc(pr, t))
        # hand direct path (final-position edges only): contrib through frozen final norm + unembed, mean-centred
        direct = None
        if pos == last:
            fin = cap0["final_in"][0, last]
            scale = torch.rsqrt(fin.pow(2).mean() + eps)
            y = contrib * scale * (1.0 + model.model.norm.weight)
            dl = y @ WU.T
            direct = (dl[t] - dl.mean()).item()
        W = W_of.get(nid)
        r = dict(
            graph=g["metadata"]["slug"],
            node=nid,
            tok_ok=tok_ok,
            target=t,
            target_str=tok.decode([t]),
            W=W,
            act_graph=ga,
            act=a_nogain,
            act_withgain=a_gain,
            act_raw=a_raw,
            ident_ok=ident_ok,
            E=E,
            E_postcap=E_post,
            dprob=dprob,
            base_prob=base["prob"],
            graph_prob=base["graph_prob"],
            base_top1_is_target=base["top1"] == t,
            C_R_mean=float(np.mean(rE)),
            C_R_max=float(np.max(rE)),
            E_over_W=(E / W if W else None),
            direct=direct,
            direct_over_W=(direct / W if (direct is not None and W) else None),
            contrib_norm=nrm.item(),
        )
        rows.append(r)
        print(json.dumps(r), flush=True)

json.dump(rows, open(f"{OUT}/results.json", "w"), indent=1)
print("TC files:", {L: v["path"] for L, v in _tc.items()})
