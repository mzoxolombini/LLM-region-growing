"""Figures/tables for the full run."""
import glob, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RUN, OUT = "runs/full", "paper_figures"
import os
os.makedirs(OUT, exist_ok=True)
SC = ["indian_pines", "pavia_university", "salinas", "botswana", "houston"]
plt.rcParams.update({"font.family": "serif", "font.size": 8})
des = sorted((json.load(open(f)) for f in glob.glob(f"{RUN}/designs/*.json")), key=lambda e: e["id"])
sel = json.load(open(f"{RUN}/selection.json"))
mk = {"D0": ("k", "s"), "E1": ("#1f77b4", "o"), "E2": ("#2ca02c", "^"), "M1": ("#d62728", "D"), "M2": ("#9467bd", "v")}
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
for op, (c, m) in mk.items():
    xs = [e["gen"] for e in des if e["op"] == op]; ys = [e["fitness"] for e in des if e["op"] == op]
    ax[0].scatter(np.array(xs) + {"D0": 0, "E1": -.15, "E2": -.05, "M1": .05, "M2": .15}[op], ys, c=c, marker=m, s=14, label=op)
best = np.maximum.accumulate([max(e["fitness"] for e in des if e["gen"] == g) for g in range(11)])
ax[0].plot(range(11), best, "k-", lw=1, label="best so far")
ax[0].set_ylim(85, 100.5); ax[0].set_xlabel("generation"); ax[0].set_ylabel("fitness $F$")
ax[0].legend(fontsize=6, ncol=3, loc="center right"); ax[0].set_xticks(range(11))
low = [e for e in des if e["fitness"] < 85]
ax[0].text(0.03, 0.03, f"{len(low)} designs below 85 not shown", transform=ax[0].transAxes, fontsize=6, va="bottom")
ok = [e for e in des if e["fitness"] > 0]
sz = [np.mean([e["per_scene"][s]["mean_region_size"] for s in SC]) for e in ok]
ax[1].scatter(sz, [e["fitness"] for e in ok], c=[e["gen"] for e in ok], cmap="viridis", s=14)
for key, lab in (("LLM-MRG", "LLM-MRG"), ("LLM-RB", "LLM-RB"), ("Baseline MRG", "D0")):
    e = next(x for x in ok if x["id"] == sel[key]["id"])
    ax[1].annotate(lab, (np.mean([e["per_scene"][s]["mean_region_size"] for s in SC]), e["fitness"]),
                   fontsize=6, xytext=(4, -8), textcoords="offset points")
ax[1].set_xscale("log"); ax[1].set_xlabel("mean region size (pixels)"); ax[1].set_ylabel("fitness $F$")
ax[1].set_ylim(85, 100.5); ax[1].set_xlim(1, 50)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 10)); cb = fig.colorbar(sm, ax=ax[1]); cb.set_label("generation")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_search.pdf"); plt.close(fig)
# per-generation table
rows = []
for g in range(11):
    gs = [e for e in des if e["gen"] == g]
    b = max(gs, key=lambda e: e["fitness"])
    nf = sum(1 for e in gs if e["fitness"] == 0)
    rows.append((g, b["op"], b["fitness"], np.mean([b["per_scene"][s]["oa_grp"] for s in SC]),
                 np.mean([b["per_scene"][s]["mean_region_size"] for s in SC]), nf, b["description"]))
lines = [f"{g} & {op} & {f:.2f} & {grp:.2f} & {sz:.1f} & {nf} \\\\" for g, op, f, grp, sz, nf, _ in rows]
open(f"{OUT}/tab_gen.tex", "w").write("\n".join(lines) + "\n")
for r in rows: print(r[0], r[1], round(r[2], 2), round(r[3], 2), round(r[4], 1), r[5], r[6][:90])
print("failed designs:", [(e["id"], list(e["per_scene"].values())[0].get("error", "")[:80]) for e in des if e["fitness"] == 0])
