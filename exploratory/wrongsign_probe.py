"""POST HOC, exploratory: where does the effect of removing a feature get undone?
For chosen edges: dose-response of the centred pre-cap target logit, and the layer-by-layer running
effect (change in residual stream at the final position, read through the frozen final norm and unembedding)."""
import json, os, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = "/mnt/d/AI/cache/huggingface/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"
L0 = [76, 65, 49, 54, 88, 87, 95, 70, 52, 72, 88, 5, 6, 8, 8, 8, 10, 12, 13, 12, 11, 13, 15, 25, 37, 41]
EDGES = [("graphs2/gemma-bee-insect", n) for n in ("22_16038_6", "14_11504_2", "19_5460_6")] + [
    ("graphs2/gemma-dollar", "18_10102_6"), ("graphs2/gemma-dollar", "20_5796_6"), ("graphs2/gemma-english", "19_13366_6"),
    ("graphs2/gemma-small-big-zh", "23_8488_8"), ("graphs2/gemma-verano", "24_13490_9"), ("graphs2/gemma-G", "23_14585_7"),
    ("graphs/gemma-fact-dallas-austin", "21_5943_10"), ("graphs/gemma-fact-dallas-austin", "20_15589_10"), ("graphs2/gemma-girls-are", "22_11517_6")]
dev = "cuda"
tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
st = {"L": None, "edit": None, "pos": -1, "x": None, "precap": None}
def mk(L):
    lay = model.model.layers[L]
    def pre(m, a):
        if st["L"] == L:
            r = a[0][0, st["pos"]].detach(); st["x"] = r * torch.rsqrt(r.pow(2).mean() + 1e-6)
    def post(m, a, o):
        if st["L"] != L or st["edit"] is None: return o
        o = o.clone(); o[0, st["pos"]] += st["edit"]; return o
    lay.pre_feedforward_layernorm.register_forward_pre_hook(pre); lay.post_feedforward_layernorm.register_forward_hook(post)
for L in range(26): mk(L)
model.model.norm.register_forward_pre_hook(lambda m, a: st.__setitem__("final", a[0][0, -1].detach()))
model.lm_head.register_forward_hook(lambda m, a, o: st.__setitem__("precap", o[0, -1].detach()))
@torch.no_grad()
def run(ids, L, pos, edit=None):
    st["L"], st["pos"], st["edit"] = L, pos, edit
    out = model(ids, output_hidden_states=True)
    return st["precap"].clone(), [h[0, -1].clone() for h in out.hidden_states[:-1]] + [st["final"].clone()], torch.softmax(out.logits[0, -1], -1)
res = []
with torch.no_grad():
    for gp, node in EDGES:
        g = json.load(open(f"{HERE}/{gp}.json")); L, f, pos = map(int, node.split("_"))
        ids = tok(g["metadata"]["prompt"].removeprefix("<bos>"), return_tensors="pt").input_ids.to(dev); p = pos - ids.shape[1]
        lg = sorted([n for n in g["nodes"] if n["feature_type"] == "logit"], key=lambda n: -n["token_prob"]); tgt = int(lg[0]["node_id"].split("_")[1])
        W = next(l["weight"] for l in g["links"] if l["source"] == node and l["target"] == lg[0]["node_id"])
        P = np.load(f"/mnt/ai/models/gemma-scope-transcoders/layer_{L}/width_16k/average_l0_{L0[L]}/params.npz")
        pc0, h0, pr0 = run(ids, L, p)
        pre_ = st["x"] @ torch.tensor(P["W_enc"], device=dev)[:, f] + float(P["b_enc"][f]); act = pre_.item() if pre_ > float(P["threshold"][f]) else 0.0
        contrib = act * torch.tensor(P["W_dec"][f], device=dev)
        cen = lambda v: (v[tgt] - v.mean()).item()
        scale = torch.rsqrt(h0[-1].pow(2).mean() + 1e-6) * (1 + model.model.norm.weight)
        lens = lambda dh: cen((dh * scale) @ model.lm_head.weight.T)
        dose = {}
        for a in (0.1, 0.25, 0.5, 1.0):
            pc, h, pr = run(ids, L, p, -a * contrib)
            dose[a] = dict(E=cen(pc0) - cen(pc), dp=(pr[tgt] - pr0[tgt]).item())
            if a == 1.0: running = [-lens(h[k] - h0[k]) for k in range(len(h))]  # drop attributable up to each layer output
        row = dict(graph=gp.split("/")[1], node=node, W=W, act=act, p0=pr0[tgt].item(), dose=dose,
                   running_from_layer={k - 1: round(running[k], 2) for k in range(L + 1, 27)})
        res.append(row); print(json.dumps(row), flush=True)
json.dump(res, open(f"{HERE}/results/wrongsign-probe.json", "w"), indent=1)
