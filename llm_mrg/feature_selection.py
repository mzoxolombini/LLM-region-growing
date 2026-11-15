"""Feature-selection variants of the baseline (Section 6.3.3): Baseline MRG,
MRG-PSO and MRG-ASVM share the same regions and features (88 GLCM + mean
spectrum) and differ only in which features the SVM uses.

  MRG-PSO  : binary PSO with a sigmoid transfer function
  MRG-ASVM : binary grasshopper optimisation algorithm (GOA, Saremi et al. 2017)

Wrapper fitness (ASSUMED, Chapter 6 not supplied):
  f = alpha * (1 - CV accuracy) + (1 - alpha) * (selected / total)
with a stratified 3-fold CV of a linear SVM (liblinear) on at most 1500 training
regions; the selected subset is then used by the harness SVM on all regions.
Only training-split labels are used.

"Adaptive" in ASVM is not defined in the chapter supplied. If it refers to
adapting SVM hyperparameters as well, extend the GOA position vector with
log C and log gamma.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


MAX_FS_SAMPLES = 1500  # training regions used inside the wrapper fitness (speed)


def _subsample(X, y, seed, n=MAX_FS_SAMPLES):
    if len(y) <= n:
        return X, y
    from sklearn.model_selection import train_test_split
    _, cnt = np.unique(y, return_counts=True)
    strat = y if cnt.min() >= 2 else None
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=strat, random_state=seed)
    return Xs, ys


def _fitness(mask, X, y, alpha, folds, seed):
    if mask.sum() == 0:
        return 1.0
    _, cnt = np.unique(y, return_counts=True)
    cv = (StratifiedKFold(folds, shuffle=True, random_state=seed) if cnt.min() >= folds
          else KFold(folds, shuffle=True, random_state=seed))
    clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto", max_iter=500))
    acc = cross_val_score(clf, X[:, mask], y, cv=cv).mean()
    return alpha * (1 - acc) + (1 - alpha) * mask.mean()


def binary_pso(X, y, n_particles=10, iters=20, alpha=0.99, folds=3, seed=0,
               w=0.9, c1=2.0, c2=2.0, vmax=6.0):
    X, y = _subsample(X, y, seed)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    pos = rng.random((n_particles, d)) < 0.5
    vel = rng.uniform(-1, 1, (n_particles, d))
    fit = np.array([_fitness(p, X, y, alpha, folds, seed) for p in pos])
    pbest, pfit = pos.copy(), fit.copy()
    g = pbest[pfit.argmin()].copy()
    gfit = pfit.min()
    HISTORY.clear(); HISTORY.append(float(gfit))
    for t in range(iters):
        wt = w - (w - 0.4) * t / max(iters - 1, 1)
        r1, r2 = rng.random((2, n_particles, d))
        vel = np.clip(wt * vel + c1 * r1 * (pbest ^ pos) * np.where(pbest, 1, -1)
                      + c2 * r2 * (g ^ pos) * np.where(g, 1, -1), -vmax, vmax)
        pos = rng.random((n_particles, d)) < 1 / (1 + np.exp(-vel))
        fit = np.array([_fitness(p, X, y, alpha, folds, seed) for p in pos])
        imp = fit < pfit
        pbest[imp], pfit[imp] = pos[imp], fit[imp]
        if pfit.min() < gfit:
            gfit, g = pfit.min(), pbest[pfit.argmin()].copy()
        HISTORY.append(float(gfit))
    return g, gfit


def binary_goa(X, y, n_particles=10, iters=20, alpha=0.99, folds=3, seed=0,
               cmax=1.0, cmin=1e-5, f=0.5, l=1.5):
    """Grasshopper optimisation in [0,1]^d, binarised with a sigmoid transfer."""
    X, y = _subsample(X, y, seed)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    lb, ub = 0.0, 1.0
    pos = rng.random((n_particles, d))
    s = lambda r: f * np.exp(-r / l) - np.exp(-r)
    to_bin = lambda P: rng.random(P.shape) < 1 / (1 + np.exp(-10 * (P - 0.5)))
    bins = to_bin(pos)
    fit = np.array([_fitness(b, X, y, alpha, folds, seed) for b in bins])
    best, bfit = bins[fit.argmin()].copy(), fit.min()
    target = pos[fit.argmin()].copy()
    HISTORY.clear(); HISTORY.append(float(bfit))
    for t in range(iters):
        c = cmax - t * (cmax - cmin) / max(iters - 1, 1)
        new = np.empty_like(pos)
        for i in range(n_particles):
            diff = pos - pos[i]
            dist = np.linalg.norm(diff, axis=1)
            dist[i] = 1.0
            rn = 2 + np.mod(dist, 2)  # map distances into [2, 4] as in Saremi et al.
            social = (c * (ub - lb) / 2 * s(rn)[:, None] * diff / dist[:, None])
            social[i] = 0
            new[i] = np.clip(c * social.sum(0) + target, lb, ub)
        pos = new
        bins = to_bin(pos)
        fit = np.array([_fitness(b, X, y, alpha, folds, seed) for b in bins])
        if fit.min() < bfit:
            bfit, best, target = fit.min(), bins[fit.argmin()].copy(), pos[fit.argmin()].copy()
        HISTORY.append(float(bfit))
    return best, bfit


HISTORY: list = []  # best wrapper fitness after each iteration of the last run

SELECTORS = {"pso": binary_pso, "goa": binary_goa}
