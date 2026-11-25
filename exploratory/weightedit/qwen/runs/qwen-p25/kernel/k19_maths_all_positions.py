"""Read pass (original), ALL positions (prompt included): 100 maths texts and 100 code texts. Where do the two neurons fire?"""
import json, sys, numpy as np, collections
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
Q, C = H.Q, H.C
tok = H.model()["tok"]
B = json.load(open(f"{C.DATA}/runs/qwen-baseline-run1.json")); M = Q.load_maths()
r = {a["task_id"]: a for a in B["A"]}
P = [p for p in Q.load_practice() if p["task_id"] <= 800][:100]
T = [dict(user=M[i]["question"] + Q.MATHS_TAIL, reply=B["M"][i]["reply"]) for i in range(100)] + [dict(user=Q.mbpp_user(p), reply=r[p["task_id"]]["reply"]) for p in P]
path = H.read_pass(T, ["mlp_in14", "mlp_in23"], "original", reply_only=False)
Z = np.load(path); E = H.current_edit()
for L in (14, 23):
    G, U = E[f"model.layers.{L}.mlp.gate_proj.weight"], E[f"model.layers.{L}.mlp.up_proj.weight"]
    N = int((G != H.original(f"model.layers.{L}.mlp.gate_proj.weight")).any(1).nonzero()[0])
    x = Z[f"mlp_in{L}"].astype(np.float32); g = x @ G[N].numpy(); act = g / (1 + np.exp(-np.clip(g, -60, 60))) * (x @ U[N].numpy())
    np.save(path.replace(".npz", f"_act{L}.npy"), act)
    for nm, sel in (("maths", Z["text"] < 100), ("code", Z["text"] >= 100)):
        f = (act > 0.5) & sel
        print(f"layer {L} {nm}: fires on {int(f.sum())} positions in {len(set(Z['text'][f].tolist()))} texts; by position index: {collections.Counter(int(p) for p in Z['pos'][f]).most_common(5)}; tokens {collections.Counter(tok.decode([t]) for t in Z['token'][f]).most_common(6)}; mean |act| off {np.abs(act[sel & ~f]).mean():.2e}")
