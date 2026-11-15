"""Merge original and revision results; tables and tests for the revised paper."""
import glob, gzip, json, os, sys
from collections import defaultdict
import numpy as np


def _load(f):
    """Job files are stored gzipped in the repository (*.json.gz)."""
    import gzip, json
    return json.load(gzip.open(f, "rt")) if f.endswith(".gz") else json.load(open(f))

from scipy import stats

RUN, OUT = "runs/full", "paper_tables"
os.makedirs(OUT, exist_ok=True)
SC = ["indian_pines", "pavia_university", "salinas", "botswana", "houston"]
SCN = dict(zip(SC, ["IP", "PU", "SA", "BO", "HU"]))
METH = ["Baseline MRG", "MRG-PSO", "MRG-ASVM", "DoE", "EPF", "SVM-PC", "PIX-MRG", "LLM-MRG", "PIX-RB", "LLM-RB"]
HEAD = ["Baseline", "PSO", "ASVM", "DoE", "EPF", "PIX-MRG", "LLM-MRG", "PIX-RB", "LLM-RB"]
R = []
for f in glob.glob(f"{RUN}/final/jobs/*.json*"):
    for r in _load(f):
        if r["method"] not in ("MRG-PSO", "MRG-ASVM"):
            R.append(r)
for f in glob.glob(f"{RUN}/final/jobs_v2/*.json*") + glob.glob(f"{RUN}/final/jobs_v3/*.json*"):
    R += _load(f)
for r in R:
    r["protocol"], r["seed"] = r["tag"].split("_")[0], int(r["tag"].split("_")[1])
    r.pop("pred", None)
G = defaultdict(list)
for r in R:
    G[(r["protocol"], r["scene"], r["method"])].append(r)
json.dump(R, open(f"{OUT}/all_results.json", "w"))

def vals(p, s, m, k="oa", seeds=None):
    sc = 100.0 if k == "kappa" else 1.0
    return sc * np.array([x[k] for x in sorted(G.get((p, s, m), []), key=lambda x: x["seed"])
                     if x[k] is not None and (seeds is None or x["seed"] in seeds)], float)

def fmt_row(cells, nd=2, sd=True):
    means = [c.mean() if c.size else None for c in cells]
    best = max(round(m, nd) for m in means if m is not None)
    out = []
    for c, m in zip(cells, means):
        if m is None:
            out.append("--"); continue
        t = f"{m:.{nd}f}" + (f"$\\pm${c.std(ddof=1):.{nd}f}" if sd and c.size > 1 else "")
        out.append(f"\\textbf{{{t}}}" if round(m, nd) == best else t)
    return out

def table(p, k="oa", sd=True, seeds=None, mean_row=True):
    L = []
    for s in SC:
        L.append(f"{SCN[s]} & " + " & ".join(fmt_row([vals(p, s, m, k, seeds) for m in METH], sd=sd)) + " \\\\")
    if mean_row:
        cells = [np.array([vals(p, s, m, k, seeds).mean() for s in SC if vals(p, s, m, k, seeds).size]) for m in METH]
        cells = [c if c.size == 5 else np.array([]) for c in cells]
        L.append("\\midrule\nMean & " + " & ".join(fmt_row(cells, sd=False)) + " \\\\")
    return "\n".join(L)

T = {}
T["random"] = table("random")
T["split0"] = table("random", sd=False, seeds={0})
T["aa"] = table("random", "aa", sd=False)
T["kappa"] = table("random", "kappa", sd=False)
T["ns"] = table("random", "oa_regions_without_train_pixels", sd=False)
T["spatial"] = table("spatial")
T["buffered"] = table("buffered")
T["low"] = table("low")
T["tiles64"] = table("tiles64")
for k, v in T.items():
    open(f"{OUT}/tab_{k}.tex", "w").write(v + "\n")

# corrected resampled t-test (Nadeau & Bengio 2003) per scene, random protocol, 10 splits
def nb(a, b, ratio=0.3 / 0.7):
    d = a - b
    J = d.size
    v = d.var(ddof=1)
    if v == 0:
        return float(np.sign(d.mean()) * np.inf), 0.0 if d.mean() != 0 else 1.0
    t = d.mean() / np.sqrt((1 / J + ratio) * v)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))

PAIRS = [("LLM-MRG", "Baseline MRG"), ("LLM-MRG", "DoE"), ("LLM-MRG", "MRG-PSO"), ("LLM-MRG", "MRG-ASVM"),
         ("LLM-MRG", "PIX-MRG"), ("LLM-MRG", "EPF"), ("LLM-RB", "PIX-RB"), ("LLM-RB", "LLM-MRG"), ("LLM-RB", "EPF")]
nbt = []
for a, b in PAIRS:
    row = [f"{a} vs {b}"]
    for s in SC:
        x, y = vals("random", s, a), vals("random", s, b)
        t, p = nb(x, y)
        row.append((round(x.mean() - y.mean(), 2), p))
    nbt.append(row)
L = []
for row in nbt:
    cells = []
    for d, p in row[1:]:
        star = "$^{*}$" if p < 0.05 else ""
        cells.append(f"{d:+.2f} ({p:.3f}){star}" if p >= 0.001 else f"{d:+.2f} ($<$0.001){star}")
    L.append(row[0].replace("Baseline MRG", "Baseline").replace("MRG-PSO", "PSO").replace("MRG-ASVM", "ASVM") + " & " + " & ".join(cells) + " \\\\")
open(f"{OUT}/tab_nb.tex", "w").write("\n".join(L) + "\n")

# Friedman over scenes (blocks) on per-scene mean OA, per protocol
fr = {}
for p in ("random", "spatial", "tiles64", "buffered", "low"):
    M = [m for m in METH if all(vals(p, s, m).size for s in SC)]
    X = np.array([[vals(p, s, m).mean() for m in M] for s in SC])
    st = stats.friedmanchisquare(*X.T)
    ranks = np.mean([stats.rankdata(-row) for row in X], 0)
    k = len(M); q = {10: 3.164, 9: 3.102, 8: 3.031, 7: 2.949}.get(k)
    cd = q * np.sqrt(k * (k + 1) / (6 * 5)) if q else None
    fr[p] = {"methods": M, "chi2": float(st.statistic), "p": float(st.pvalue),
             "mean_ranks": dict(zip(M, map(float, ranks))), "nemenyi_cd": cd}
json.dump({"friedman": fr, "nb": nbt}, open(f"{OUT}/tests_v2.json", "w"), indent=1, default=str)
print(json.dumps(fr, indent=1))
for row in nbt: print(row)
# test-set sizes for buffered
for s in SC:
    n = [len(x["per_class_acc"]) for x in G[("buffered", s, "LLM-RB")]]
    print(s, "buffered test classes per seed", n)

# gaps (leads) per protocol, mean over scenes, and per-scene corrected tests for buffered/tiles64
gaps = {}
for p in ("random", "spatial", "tiles64", "buffered", "low"):
    g = {}
    for a, b in (("LLM-RB", "Baseline MRG"), ("LLM-RB", "DoE"), ("LLM-MRG", "Baseline MRG"), ("LLM-MRG", "DoE"),
                 ("PIX-RB", "SVM-PC"), ("PIX-MRG", "SVM-PC"), ("DoE", "Baseline MRG")):
        g[f"{a} - {b}"] = [round(float(vals(p, s_, a).mean() - vals(p, s_, b).mean()), 2) for s_ in SC]
        g[f"{a} - {b}"].append(round(float(np.mean(g[f"{a} - {b}"])), 2))
    gaps[p] = g
json.dump(gaps, open(f"{OUT}/gaps.json", "w"), indent=1)
for p, g in gaps.items():
    print(p)
    for k, v in g.items(): print("   ", k, v)
