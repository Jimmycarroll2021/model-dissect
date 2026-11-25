"""Checks of the harness in a throwaway run directory (uses 3 passes of a scratch budget; touches no held-out data)."""
import os, sys, tempfile, json, numpy as np, torch
d = tempfile.mkdtemp(dir="/mnt/ai/model-dissect-data/weightedit/runs", prefix="qcheck-")
os.environ["QWEN_RUN"], os.environ["QWEN_BUDGET"] = d, "3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qharness as H
Q = H.Q

def refused(f, *a, **k):
    try:
        f(*a, **k)
    except H.Refused:
        return True
    return False

name = "model.layers.23.mlp.down_proj.weight"
o = H.original(name)
ck = lambda label, ok: (print("PASS" if ok else "FAIL", label), ok)[1]
R = []
R.append(ck("embedding refused", refused(H.save_edit, "model.embed_tokens.weight", H.original("model.embed_tokens.weight") + 1, "x", "y")))
R.append(ck("final norm refused", refused(H.save_edit, "model.norm.weight", H.original("model.norm.weight") + 1, "x", "y")))
R.append(ck("whole-tensor scaling refused", refused(H.save_edit, name, o * 1.1, "x", "y")))
R.append(ck("edit without evidence refused", refused(H.save_edit, name, o + 1, "x", "")))
R.append(ck("held-back problem refused", refused(H.score_pass, [("mbpp", 801)], "original")))
R.append(ck("no pass spent by refusals", H._state()["passes_used"] == 0))
# pass 1: unedited answer equals the plain model's answer
P = Q.load_practice()[0]
want = Q.run_mbpp(H.model(), P)
summ, res = H.score_pass([("mbpp", P["task_id"])], "original")
R.append(ck("unedited pass reproduces the plain answer", res["original"][0] == want))
# pass 2: a destructive edit changes the answer, and the model is restored afterwards
t = o.clone(); t[:, :] = 0.0; t[0, 0] = 1.0
H.save_edit(name, t, "check", "check")
summ, res = H.score_pass([("mbpp", P["task_id"])], "both")
R.append(ck("edit changes the answer", res["edited"][0]["reply"] != res["original"][0]["reply"]))
R.append(ck("original side unchanged", res["original"][0] == want))
R.append(ck("weights restored after the pass", torch.equal(H._params[name].detach().cpu(), o)))
# pass 3: read pass shapes
p = H.read_pass([dict(user=Q.mbpp_user(P), reply=want["reply"])], ["resid_pre0", "mlp_in5", "mlp_act23", "attn_out3", "attn_out4", "mlp_out7", "resid_final"])
Z = np.load(p); n = len(Z["token"])
R.append(ck("read pass shapes", all(Z[k].shape == (n, w) for k, w in (("resid_pre0", 1024), ("mlp_in5", 1024), ("mlp_act23", 3584), ("attn_out3", 1024), ("attn_out4", 1024), ("mlp_out7", 1024), ("resid_final", 1024)))))
R.append(ck("teacher-forced predictions match the greedy reply", (Z["pred"][:-1] == Z["target"][:-1]).mean() > 0.98))
R.append(ck("budget enforced", refused(H.score_pass, [("mbpp", P["task_id"])], "original")))
man = H.freeze()
E = H.apply_edit(d)
R.append(ck("freeze and rebuild are exact", torch.equal(E[name], t)))
R.append(ck("frozen run refuses edits", refused(H.save_edit, name, o + 1, "x", "y")))
print("ALL PASS" if all(R) else "SOME FAILED", d)
