"""LaTeX tables for the test-set results."""
import json, sys
from collections import defaultdict
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/main"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper_figures"
import os
os.makedirs(OUT, exist_ok=True)
SC = ["indian_pines", "pavia_university", "salinas", "botswana", "houston"]
SCN = {"indian_pines": "IP", "pavia_university": "PU", "salinas": "SA", "botswana": "BO", "houston": "HU"}
METH = ["Baseline MRG", "MRG-PSO", "MRG-ASVM", "DoE", "LLM-MRG", "LLM-RB"]
R = json.load(open(f"{RUN}/final/results.json"))
T = json.load(open(f"{RUN}/final/tests.json"))
g = defaultdict(list)
for r in R:
    g[(r["protocol"], r["scene"], r["method"])].append(r)

def cell(p, s, m, k, scale=1.0, nd=2):
    v = [x[k] for x in g.get((p, s, m), []) if x[k] is not None]
    if not v:
        return "--", None
    v = np.array(v) * scale
    txt = f"{v.mean():.{nd}f}" + (f"$\\pm${v.std(ddof=1):.{nd}f}" if len(v) > 1 else "")
    return txt, v.mean()

def bold_row(vals):
    best = max((x[1] for x in vals if x[1] is not None), default=None)
    return [f"\\textbf{{{t}}}" if (m is not None and round(m, 2) == round(best, 2)) else t for t, m in vals]

lines = []
for s in SC:
    for k, lab, sc_, nd in (("oa", "OA", 1, 2), ("aa", "AA", 1, 2), ("kappa", "$\\kappa\\times100$", 100, 2)):
        vals = [cell("random", s, m, k, sc_, nd) for m in METH]
        first = f"\\multirow{{3}}{{*}}{{{SCN[s]}}}" if k == "oa" else ""
        lines.append(f"{first} & {lab} & " + " & ".join(bold_row(vals)) + " \\\\")
    lines.append("\\midrule" if s != SC[-1] else "")
# mean over scenes of OA
vals = []
for m in METH:
    v = [np.mean([x["oa"] for x in g[("random", s, m)]]) for s in SC if g.get(("random", s, m))]
    vals.append((f"{np.mean(v):.2f}", np.mean(v)) if len(v) == 5 else ("--", None))
lines.append("\\midrule\nMean & OA & " + " & ".join(bold_row(vals)) + " \\\\")
open(f"{OUT}/tab_test.tex", "w").write("\n".join(l for l in lines if l is not None) + "\n")

# leakage table: OA_ns (random) and spatial OA
L = []
for s in SC:
    a = [cell("random", s, m, "oa_regions_without_train_pixels") for m in METH]
    b = [cell("spatial", s, m, "oa") for m in METH]
    f = [cell("random", s, m, "frac_test_in_regions_with_train_pixels", 100, 1) for m in METH]
    L.append(f"\\multirow{{3}}{{*}}{{{SCN[s]}}} & shared (\\%) & " + " & ".join(t for t, _ in f) + " \\\\")
    L.append(f" & $OA_{{ns}}$ & " + " & ".join(bold_row(a)) + " \\\\")
    L.append(f" & $OA_{{sp}}$ & " + " & ".join(bold_row(b)) + " \\\\")
    if s != SC[-1]:
        L.append("\\midrule")
open(f"{OUT}/tab_leak.tex", "w").write("\n".join(L) + "\n")

# McNemar summary
mc = defaultdict(lambda: [0, 0, 0])  # better, worse, n.s.
for t in T["mcnemar"]:
    if t["protocol"] != "random":
        continue
    key = t["b"]
    if t["p"] < 0.05:
        mc[key][0 if t["n10"] < t["n01"] else 1] += 1
    else:
        mc[key][2] += 1
print("McNemar (ref better, ref worse, n.s.):", dict(mc))
print("Wilcoxon:", T["wilcoxon_oa"]); print("Friedman:", T["friedman_oa"])
json.dump({k: v for k, v in mc.items()}, open(f"{OUT}/mcnemar_summary.json", "w"))
print(open(f"{OUT}/tab_test.tex").read()); print(open(f"{OUT}/tab_leak.tex").read())
