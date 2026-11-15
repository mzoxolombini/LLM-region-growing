"""Figures and LaTeX tables for the paper from runs/main."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/main"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper_figures"
import os
os.makedirs(OUT, exist_ok=True)
SC = ["indian_pines", "pavia_university", "salinas", "botswana", "houston"]
SCN = ["IP", "PU", "SA", "BO", "HU"]
plt.rcParams.update({"font.family": "serif", "font.size": 8})

des = [json.load(open(f)) for f in sorted(glob.glob(f"{RUN}/designs/*.json"))]
des.sort(key=lambda e: (e["gen"], e["id"]))
sel = json.load(open(f"{RUN}/selection.json"))

# Fig: per-scene fitness heatmap of every design
M = np.array([[e["per_scene"][s]["fitness"] for s in SC] for e in des])
lab = [f"{e['id'][:6]} {e['op']}" for e in des]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
im = ax.imshow(M, cmap="viridis", vmin=50, vmax=100, aspect="auto")
ax.set_xticks(range(5), SCN); ax.set_yticks(range(len(lab)), lab)
for i in range(M.shape[0]):
    for j in range(5):
        ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=6.5,
                color="w" if M[i, j] < 80 else "k")
cb = fig.colorbar(im, ax=ax); cb.set_label("$F_d$")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_search.pdf"); plt.close(fig)

# Table: search archive
rows = []
for e in des:
    tag = ""
    if sel.get("LLM-MRG") and e["id"] == sel["LLM-MRG"]["id"]: tag = " (LLM-MRG)"
    if sel.get("LLM-RB") and e["id"] == sel["LLM-RB"]["id"]: tag = " (LLM-RB)"
    ps = e["per_scene"]
    mean = lambda k: np.mean([ps[s][k] for s in SC]) if all(ps[s].get(k) is not None for s in SC) else float("nan")
    rows.append(f"{e['id'][:6].replace('_', '\\_')} & {e['op']} & {mean('oa_val'):.2f} & {mean('oa_grp'):.2f} & "
                f"{mean('mf1_val'):.3f} & {mean('mean_region_size'):.1f} & {e['fitness']:.2f}{tag} \\\\")
open(f"{OUT}/tab_search.tex", "w").write("\n".join(rows) + "\n\\bottomrule\n")
print("\n".join(rows))
