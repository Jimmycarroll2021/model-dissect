"""Generate report-only figure from saved producer metrics; no model imports/calls."""
from pathlib import Path
import json, hashlib, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "evidence/transfer-003/delivery/producer-report-metrics-v1.json"
data = json.loads(INPUT.read_text())
assert len(data["heldout"]) == 16 and all(x["eligible"] for x in data["heldout"])
plt.rcParams.update({"font.size": 10, "svg.hashsalt": "gpt2-producer-v1"})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
colors = ["#226e9c", "#337c50"]
for ax, key, title, color in zip(axes, ["P", "Q"], ["Bidirectional recovery P", "Advantage over orthogonal control Q"], colors):
    values = [x["native"][key] for x in data["heldout"]]
    med = statistics.median(values)
    assert abs(med-data["summary"]["median_"+key]) < 1e-12
    ax.scatter(range(1,17), values, color=color, s=35, label="Each held-out pair", zorder=3)
    ax.axhline(med, color=color, linewidth=1.8, label=f"Median {med:.4f}")
    threshold = data["thresholds"]["median_"+key]
    ax.axhline(threshold, color="#ae3d34", linestyle="--", label=f"Required median {threshold:.2f}")
    ax.axhline(0, color="#777777", linewidth=.7)
    ax.set(title=title, xlabel="Held-out pair", ylabel="Normalized contrast effect", ylim=(-0.015, .225), xticks=[1,4,8,12,16])
    ax.grid(axis="y", alpha=.15); ax.legend(loc="upper right", fontsize=8)
fig.suptitle("GPT-2 layer 8, head 8: weak positive effects below frozen criteria", fontsize=13)
fig.text(.5,.025,"Producer result only; independent scientific repeat pending. 16/16 qualified pairs. Threshold lines apply to medians.",ha="center",fontsize=9)
fig.tight_layout(rect=[0,.06,1,.93])
outputs=[]
for ext in ["png","svg"]:
    p=ROOT / f"docs/figures/gpt2-heldout-producer-v1.{ext}"
    fig.savefig(p,dpi=160,metadata={"Date":None} if ext=="svg" else {})
    outputs.append({"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
receipt={"input_sha256":hashlib.sha256(INPUT.read_bytes()).hexdigest(),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"outputs":outputs,"model_calls":0,"checks":"16 eligible rows; medians independently recomputed from saved metric rows agree with summary within1e-12"}
(ROOT/"evidence/transfer-003/delivery/producer-chart-v1.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
