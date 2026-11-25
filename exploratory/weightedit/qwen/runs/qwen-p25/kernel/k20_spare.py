"""No pass: pick the neuron to overwrite by its ORIGINAL contribution over all positions (prompt + reply, maths + code; read 19),
computed from the saved MLP inputs and the original gate/up/down weights. The earlier pick looked at code replies only."""
import json, sys, numpy as np, torch
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
Z = np.load(f"{H.RUN}/reads/fwd_19.npz"); out = {}
for L, old in ((14, 248), (23, 1998)):
    pre = f"model.layers.{L}.mlp."
    G, U, D = (H.original(pre + n + ".weight").cuda() for n in ("gate_proj", "up_proj", "down_proj"))
    x = torch.tensor(Z[f"mlp_in{L}"].astype(np.float32)).cuda()
    mx = torch.zeros(G.shape[0], device="cuda"); mean = torch.zeros_like(mx)
    for i in range(0, len(x), 8192):
        a = (torch.nn.functional.silu(x[i:i+8192] @ G.T) * (x[i:i+8192] @ U.T)).abs()
        mx = torch.maximum(mx, a.max(0).values); mean += a.sum(0)
    c = D.norm(dim=0); score = mx * c
    N = int(score.argmin())
    print(f"layer {L}: old pick {old}: max contribution {score[old]:.3f} (rank {int((score < score[old]).sum())} of {len(score)}); new pick {N}: max {score[N]:.4f}, mean {(mean[N]/len(x)*c[N]):.5f}; median neuron max {score.median():.2f}")
    out[L] = N
json.dump(out, open(f"{H.RUN}/reads/k20_spare.json", "w"))
