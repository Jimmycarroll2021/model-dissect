"""Scripted harness checks (spec 005 Story 2). Uses a throwaway run directory with a budget of 2."""
import os, shutil, sys, tempfile
d = tempfile.mkdtemp(dir="/dev/shm")
os.environ["WEIGHTEDIT_RUN"], os.environ["WEIGHTEDIT_BUDGET"] = d, "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, common as C, harness as H

O = H.original()
W = "gpt_neox.layers.3.mlp.dense_4h_to_h.weight"
ok = lambda name, cond: (print(("PASS " if cond else "FAIL ") + name), cond)[1]
res = []

def refused(fn):
    try:
        fn(); return False
    except H.Refused:
        return True

# 1 no-op: an "edit" equal to the original gives identical logits
m0, m1 = H.build(), H.build({W: O[W].clone()})
ids, _ = C.encode(m0, C.mbpp_code_text(C.load_practice()[0])[0])
res.append(ok("no-op edit leaves logits identical", torch.equal(C.logits_of(m0, ids), C.logits_of(m1, ids))))
# 2 forbidden edits refused
for n in ("gpt_neox.embed_in.weight", "embed_out.weight", "gpt_neox.final_layer_norm.weight", "gpt_neox.final_layer_norm.bias"):
    res.append(ok(f"refuses {n}", refused(lambda: H.save_edit(n, O[n] + 1, "x", "x"))))
res.append(ok("refuses whole-tensor scaling", refused(lambda: H.save_edit(W, O[W] * 1.5, "x", "x"))))
res.append(ok("refuses unchanged tensor", refused(lambda: H.save_edit(W, O[W].clone(), "x", "x"))))
res.append(ok("refuses non-finite", refused(lambda: H.save_edit(W, O[W] * float("nan"), "x", "x"))))
res.append(ok("refuses missing map entry", refused(lambda: H.save_edit(W, O[W] + 1, "x", ""))))
res.append(ok("refuses tensor that requires grad", refused(lambda: H.save_edit(W, (O[W] + 1).requires_grad_(True), "x", "x"))))
res.append(ok("gradients are disabled", not torch.is_grad_enabled()))
res.append(ok("held-out files are not loadable through common", not hasattr(C, "load_jsonl") and "heldout" not in open(C.__file__).read().split("def load_practice")[1]))
# 3 a real edit changes outputs; budget enforced
t = O[W].clone(); t[:, :8] = 0
H.save_edit(W, t, "check", "check")
s = H.score_pass(["code:0-7"], "both")
res.append(ok("a real edit changes the score", s["code/edited_minus_original"]["mean"] != 0))
H.read_pass(["code:0-3"])
res.append(ok("third pass refused at budget 2", refused(lambda: H.score_pass(["code:0-7"]))))
res.append(ok("oversize pass refused", refused(lambda: H._spend("score", 513, 512)) ))
# 4 freeze, rebuild by hash, frozen run refuses more
man = H.freeze()
R = H.apply_edit()
res.append(ok("edit rebuilds bit for bit from the published weights", torch.equal(R[W], t) and man["tensors"][W]["changed"] == int((t != O[W]).sum())))
res.append(ok("frozen run refuses further edits", refused(lambda: H.save_edit(W, O[W] + 1, "x", "x"))))
shutil.rmtree(d)
print("ALL PASS" if all(res) else "SOME FAILED")
sys.exit(0 if all(res) else 1)
