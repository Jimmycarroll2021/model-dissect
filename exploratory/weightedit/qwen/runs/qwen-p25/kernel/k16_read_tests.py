"""Read pass (original): full baseline replies of 256 work problems, last-layer MLP input/activations and final residual.
Then, no pass: can the last-layer MLP input tell 'about to start my own tests' from everything else?"""
import json, re, sys, numpy as np, torch
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
Q, C = H.Q, H.C
m = H.model(); tok = m["tok"]
r = {a["task_id"]: a for a in json.load(open(f"{C.DATA}/runs/qwen-baseline-run1.json"))["A"]}
P = [p for p in Q.load_practice() if p["task_id"] <= 800][:256]
T = [dict(user=Q.mbpp_user(p), reply=r[p["task_id"]]["reply"]) for p in P]
path = H.read_pass(T, ["mlp_in23", "mlp_act23", "resid_final"], "original")
Z = np.load(path)
# label positions: target token starts the model's own tests (first top-level '#', 'assert', 'print(' or 'if __name__' line after a def)
lab = np.zeros(len(Z["token"]), bool); fence_close = np.zeros(len(Z["token"]), bool)
for ti, t in enumerate(T):
    rep = t["reply"]; k = re.search(r"^def ", rep, re.M)
    idx = np.nonzero(Z["text"] == ti)[0]
    enc = tok(rep, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    n0 = int(Z["pos"][idx[0]]) + 1          # index of first reply token in the full sequence
    start = {a: j for j, (a, b) in enumerate(enc)}
    if k:
        mt = re.search(r"^(#|assert|print\(|if __name__)", rep[k.start():], re.M)
        if mt and (k.start() + mt.start()) in start:
            j = start[k.start() + mt.start()]      # reply token j is the target of sequence position n0 + j - 1
            hit = idx[Z["pos"][idx] == n0 + j - 1]
            lab[hit] = True
    for mm in re.finditer(r"^```\s*$", rep, re.M):
        if mm.start() in start:
            fence_close[idx[Z["pos"][idx] == n0 + start[mm.start()] - 1]] = True
print(path, "tokens", len(lab), "test-start positions", int(lab.sum()), "fence-close positions", int(fence_close.sum()))
np.save(path.replace(".npz", "_lab.npy"), lab); np.save(path.replace(".npz", "_close.npy"), fence_close)
print("targets at test-start:", __import__("collections").Counter(tok.decode([x]) for x in Z["target"][lab]).most_common(6))
print("model's own prediction right at test-start:", float((Z["pred"][lab] == Z["target"][lab]).mean()))
