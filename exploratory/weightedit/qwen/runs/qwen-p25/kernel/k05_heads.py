"""No pass: at the function-name positions of read 2, which attention heads write toward the test's name token?
Direct effect only: head values -> output projection -> final norm scale at that position -> output matrix."""
import json, sys, numpy as np, torch, inspect
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
Q, C = H.Q, H.C
m = H.model(); tok = m["tok"]; mod = m["model"]
print(inspect.getsource(type(mod.model.norm).forward)[-300:])
Z = np.load(f"{H.RUN}/reads/fwd_2.npz"); meta = json.load(open(f"{H.RUN}/reads/fwd_2_meta.json"))
text, token, target, pred, lp = Z["text"], Z["token"], Z["target"], Z["pred"], Z["logp_target"]
WU = mod.lm_head.weight.detach().float(); g = mod.model.norm.weight.detach().float()
rows = []  # (global index, text, is_first_name_token)
for ti, me in enumerate(meta):
    idx = np.nonzero(text == ti)[0]
    # name tokens = targets after the last 'def' token, excluding the final '('
    toks = [tok.decode([t]) for t in token[idx]]
    d = max(i for i, t in enumerate(toks) if t.strip() == "def")
    for j in range(d, len(idx) - 1):
        rows.append((idx[j], ti, j == d))
    assert "(" in tok.decode([target[idx[-1]]]), tok.decode([target[idx[-1]]])
R = np.array([r[0] for r in rows]); first = np.array([r[2] for r in rows]); wrong_name = np.array([meta[r[1]]["own"] != meta[r[1]]["want"] for r in rows])
miss = pred[R] != target[R]
print("name positions", len(R), "mispredicted", int(miss.sum()), "| first-token positions", int(first.sum()), "mispredicted", int((miss & first).sum()))
print("mean logp target: right-name answers %.2f, wrong-name answers %.2f" % (lp[R][~wrong_name].mean(), lp[R][wrong_name].mean()))
rf = torch.tensor(Z["resid_final"][R].astype(np.float32), device="cuda")
scale = torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + 1e-6)
dirn = (WU[torch.tensor(target[R]).cuda()] - WU[torch.tensor(pred[R]).cuda()]) * g * scale   # zero where pred == target
dirc = (WU[torch.tensor(target[R]).cuda()] - WU.mean(0)) * g * scale
out = []
for L in range(24):
    lay = mod.model.layers[L]
    W = (lay.linear_attn.out_proj if hasattr(lay, "linear_attn") else lay.self_attn.o_proj).weight.detach().float()
    nh = 16 if hasattr(lay, "linear_attn") else 8
    z = torch.tensor(Z[f"attn_z{L}"][R].astype(np.float32), device="cuda")
    hd = z.shape[1] // nh
    for h in range(nh):
        o = z[:, h * hd:(h + 1) * hd] @ W[:, h * hd:(h + 1) * hd].T
        a = (o * dirn).sum(-1).cpu().numpy(); c = (o * dirc).sum(-1).cpu().numpy()
        out.append((L, h, a[miss].mean(), c[~miss].mean(), c[miss].mean(), (a[miss] > 0).mean()))
print("top heads by mean push toward (right name token - predicted token) at mispredicted positions")
print(" L  h  miss:right-pred  hit:right-mean  miss:right-mean  frac>0")
for r in sorted(out, key=lambda r: -r[2])[:14]: print("%2d %2d %8.2f %8.2f %8.2f %6.2f" % r)
print("bottom"); 
for r in sorted(out, key=lambda r: r[2])[:8]: print("%2d %2d %8.2f %8.2f %8.2f %6.2f" % r)
gap = (rf * dirn).sum(-1).cpu().numpy()
print("mean logit gap (right - predicted) at mispredicted positions: %.2f" % gap[miss].mean())
np.save(f"{H.RUN}/reads/k05_rows.npy", R)
