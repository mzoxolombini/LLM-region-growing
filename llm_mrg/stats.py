"""Statistical tests. Section 5.5.3 is not in the chapter supplied, so this
offers the usual set for paired classifier comparison:
  mcnemar        : two methods, same test pixels (exact binomial for small b+c)
  wilcoxon_pairs : paired scores across (scene, split) pairs
  friedman       : several methods ranked over (scene, split) blocks
Replace or extend to match the thesis protocol.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def mcnemar(y, p1, p2):
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    a = (p1 == y) & (p2 != y)
    b = (p1 != y) & (p2 == y)
    n01, n10 = int(a.sum()), int(b.sum())
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "stat": 0.0, "p": 1.0}
    if n < 25:
        p = min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5))
        return {"n01": n01, "n10": n10, "stat": None, "p": float(p), "test": "exact"}
    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    return {"n01": n01, "n10": n10, "stat": float(chi2), "p": float(stats.chi2.sf(chi2, 1)),
            "test": "chi2_cc"}


def wilcoxon_pairs(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.allclose(x, y):
        return {"stat": 0.0, "p": 1.0}
    r = stats.wilcoxon(x, y)
    return {"stat": float(r.statistic), "p": float(r.pvalue)}


def friedman(*cols):
    if len(cols) < 3:
        return None
    r = stats.friedmanchisquare(*cols)
    return {"stat": float(r.statistic), "p": float(r.pvalue)}
