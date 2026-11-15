"""Evaluation harness: splits, region labelling, fitness (Eq. 8.1), test metrics.

The harness owns the labels and the split. Designs only ever see ctx.

Search-time scoring (Section 8.4.5), all inside the 70% training split S_tr:
  * S_tr -> stratified 75/25 pixel split S_in / S_val -> OA_val, mF1_val
  * region-grouped hold-out: 25% of the regions that contain S_tr pixels are
    held out whole; an SVM trained on the other regions predicts them -> OA_grp
  * F_d = (OA_val + OA_grp + 100 * mF1_val) / 3,  F = mean_d F_d
The 30% test split S_te is never touched during the search.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .context import build_context_arrays, save_context
from .datasets import load_scene, make_synthetic
from .sandbox import DesignRunError, run_isolated


# --------------------------------------------------------------------------
# scenes and splits
# --------------------------------------------------------------------------
@dataclass
class Scene:
    name: str
    ctx_path: str
    gt: np.ndarray            # (H, W), 0 = unlabelled
    labels: np.ndarray = field(init=False)  # flat
    labelled: np.ndarray = field(init=False)  # flat indices of labelled pixels

    def __post_init__(self):
        self.labels = self.gt.ravel()
        self.labelled = np.flatnonzero(self.labels > 0)


def prepare_scene(name: str, data_dir: str, cache_dir: str, n_pcs: int = 10, seed: int = 0) -> Scene:
    os.makedirs(os.path.join(cache_dir, "ctx"), exist_ok=True)
    ctx_path = os.path.join(cache_dir, "ctx", f"{name}.npz")
    if name.startswith("synthetic"):
        cube, gt = make_synthetic(seed=int(name.split("_")[-1]) if "_" in name else 0)
    else:
        cube, gt = load_scene(name, data_dir)
    if not os.path.exists(ctx_path):
        save_context(ctx_path, build_context_arrays(cube, n_pcs, seed))
    return Scene(name, ctx_path, gt)


def stratified_split(scene: Scene, train_frac: float, seed: int):
    idx = scene.labelled
    y = scene.labels[idx]
    tr, te = train_test_split(idx, train_size=train_frac, stratify=y, random_state=seed)
    return np.sort(tr), np.sort(te)


def spatial_block_split(scene: Scene, train_frac: float, seed: int, tile: int = 16):
    """Spatially disjoint check (Section 5.5.5 ASSUMED form): whole square tiles
    go to train or test, filled greedily in random order until ~train_frac of
    every class's labelled pixels is in train."""
    h, w = scene.gt.shape
    rng = np.random.default_rng(seed)
    ty, tx = np.arange(h)[:, None] // tile, np.arange(w)[None, :] // tile
    tid = (ty * ((w + tile - 1) // tile) + tx).ravel()
    idx = scene.labelled
    y = scene.labels[idx]
    classes = np.unique(y)
    tiles = np.unique(tid[idx])
    rng.shuffle(tiles)
    need = {c: train_frac * np.sum(y == c) for c in classes}
    have = {c: 0 for c in classes}
    train_tiles = set()
    for t in tiles:
        m = tid[idx] == t
        cs, cnt = np.unique(y[m], return_counts=True)
        gain = sum(max(0.0, min(need[c] - have[c], n)) for c, n in zip(cs, cnt))
        if gain > 0.5 * cnt.sum():
            train_tiles.add(t)
            for c, n in zip(cs, cnt):
                have[c] += n
    in_tr = np.isin(tid[idx], list(train_tiles))
    return np.sort(idx[in_tr]), np.sort(idx[~in_tr])


def buffered_spatial_split(scene: Scene, train_frac: float, seed: int, tile: int = 64,
                           buffer: int = 20):
    """Tile split as above, then every test pixel within `buffer` pixels (Chebyshev distance)
    of a training pixel is dropped, so no square filter of radius <= buffer centred on a test
    pixel reaches a training pixel, and vice versa."""
    from scipy import ndimage
    s_tr, s_te = spatial_block_split(scene, train_frac, seed, tile)
    h, w = scene.gt.shape
    m = np.ones(h * w, bool)
    m[s_tr] = False
    d = ndimage.distance_transform_cdt(m.reshape(h, w), metric="chessboard").ravel()
    s_te = s_te[d[s_te] > buffer]
    # classes with no training pixel cannot be predicted by any method; leave them out
    s_te = s_te[np.isin(scene.labels[s_te], np.unique(scene.labels[s_tr]))]
    return s_tr, s_te


def low_label_split(scene: Scene, n_per_class: int, seed: int):
    """n_per_class training pixels per class (at most half of a small class); rest is test."""
    rng = np.random.default_rng(seed)
    tr = []
    for c in np.unique(scene.labels[scene.labelled]):
        idx = scene.labelled[scene.labels[scene.labelled] == c]
        k = min(n_per_class, len(idx) // 2)
        tr.append(rng.choice(idx, size=k, replace=False))
    tr = np.sort(np.concatenate(tr))
    return tr, np.setdiff1d(scene.labelled, tr)


# --------------------------------------------------------------------------
# classification of regions
# --------------------------------------------------------------------------
def make_svm(spec: dict, seed: int = 0) -> SVC:
    kernel = spec.get("kernel", "linear")
    C = float(spec.get("C", 1.0))
    C = C if np.isfinite(C) and C > 0 else 1.0
    gamma = spec.get("gamma", "scale")
    if not (gamma in ("scale", "auto") or (isinstance(gamma, (int, float)) and gamma > 0)):
        gamma = "scale"
    return SVC(kernel=kernel, C=C, gamma=gamma, random_state=seed)


def _fit_predict(Xtr, ytr, Xte, spec, seed):
    if len(np.unique(ytr)) == 1:
        return np.full(len(Xte), ytr[0])
    sc = StandardScaler().fit(Xtr)
    clf = make_svm(spec, seed).fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def region_majority(regions_flat, pix, labels):
    """Majority label per region over the given pixels. Returns (region_ids, label)."""
    r = regions_flat[pix].astype(np.int64)
    y = labels[pix].astype(np.int64)
    pairs, counts = np.unique(np.stack([r, y], 1), axis=0, return_counts=True)
    # highest count per region; ties go to the smaller class label
    o = np.lexsort((pairs[:, 1], -counts, pairs[:, 0]))
    pairs = pairs[o]
    first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
    return pairs[first, 0], pairs[first, 1]


class Evaluated:
    """Holds what a design produced on one scene so several scorings can reuse it."""

    def __init__(self, regions, ids, feats, labeling, clf_spec):
        self.regions = regions.ravel()
        self.ids = np.asarray(ids)
        self.feats = feats
        self.labeling = labeling
        self.clf = clf_spec
        if labeling == "region":
            self.row_of = -np.ones(int(self.regions.max()) + 1, np.int64)
            self.row_of[self.ids] = np.arange(self.ids.size)
        else:
            self.row_of_pix = -np.ones(self.regions.size, np.int64)
            self.row_of_pix[self.ids] = np.arange(self.ids.size)

    def predict(self, train_pix, labels, target_regions, seed, rng, max_vote=200):
        """Train on train_pix, return {region_id: predicted label} for target_regions."""
        target_regions = np.asarray(target_regions)
        if self.labeling == "region":
            rid, ry = region_majority(self.regions, train_pix, labels)
            Xtr = self.feats[self.row_of[rid]]
            pred = _fit_predict(Xtr, ry, self.feats[self.row_of[target_regions]], self.clf, seed)
            return dict(zip(target_regions.tolist(), pred.tolist()))
        # pixel_vote: pixel SVM, then majority vote inside each target region
        Xtr = self.feats[self.row_of_pix[train_pix]]
        tset = set(target_regions.tolist())
        cand = self.ids[np.isin(self.regions[self.ids], target_regions)]
        p = _fit_predict(Xtr, labels[train_pix], self.feats[self.row_of_pix[cand]], self.clf, seed)
        out = {}
        rc = self.regions[cand]
        for r in tset:
            v = p[rc == r]
            if v.size:
                vals, cnt = np.unique(v, return_counts=True)
                out[r] = vals[np.argmax(cnt)]
        return out


def _pix_pred(ev: Evaluated, mapping: dict, pix):
    lut = -np.ones(int(ev.regions.max()) + 1, np.int64)
    if mapping:
        k = np.fromiter(mapping.keys(), np.int64, len(mapping))
        lut[k] = np.fromiter(mapping.values(), np.int64, len(mapping))
    return lut[ev.regions[pix]]


def purity(regions_flat, pix, labels):
    """Eq. (5.7) ASSUMED: sum over regions of the majority-class count divided by
    the number of labelled pixels considered."""
    r = regions_flat[pix].astype(np.int64)
    y = labels[pix].astype(np.int64)
    pairs, counts = np.unique(np.stack([r, y], 1), axis=0, return_counts=True)
    best = np.zeros(int(r.max()) + 1 if r.size else 1, np.int64)
    np.maximum.at(best, pairs[:, 0], counts)
    return float(best.sum() / max(len(pix), 1))


def pixel_ids_for_vote(regions_flat, target_regions, rng, max_per_region=200, extra=None):
    """In pixel_vote mode the child is asked for features of a capped sample of
    pixels in each target region (plus any extra pixels, e.g. training pixels)."""
    m = np.isin(regions_flat, target_regions)
    pix = np.flatnonzero(m)
    r = regions_flat[pix]
    perm = rng.permutation(pix.size)
    pix, r = pix[perm], r[perm]
    order = np.argsort(r, kind="stable")
    pix, r = pix[order], r[order]
    first = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    rank = np.arange(r.size) - np.repeat(first, np.diff(np.r_[first, r.size]))
    keep = pix[rank < max_per_region]
    if extra is not None:
        keep = np.union1d(keep, extra)
    return np.sort(keep)


# --------------------------------------------------------------------------
# search-time fitness
# --------------------------------------------------------------------------
@dataclass
class SearchSplit:
    s_tr: np.ndarray
    s_in: np.ndarray
    s_val: np.ndarray


def make_search_split(scene: Scene, train_frac: float, inner_frac: float, seed: int) -> SearchSplit:
    s_tr, _ = stratified_split(scene, train_frac, seed)
    y = scene.labels[s_tr]
    s_in, s_val = train_test_split(s_tr, train_size=inner_frac, stratify=y, random_state=seed + 1)
    return SearchSplit(s_tr, np.sort(s_in), np.sort(s_val))


def score_search(src: str, scene: Scene, split: SearchSplit, cfg: dict, seed: int) -> dict:
    """Fitness of one design on one scene. Never raises; failures give 0."""
    t0 = time.time()
    labels = scene.labels
    rng = np.random.default_rng(seed + 7919)
    labeling = {}

    def request(regions, mode):
        rf = regions.ravel()
        tr_regions = np.unique(rf[split.s_tr])
        labeling["tr_regions"] = tr_regions
        if mode == "pixel_vote":
            return pixel_ids_for_vote(rf, tr_regions, rng, cfg.get("max_vote_pixels", 200), split.s_tr)
        return tr_regions

    out = {"scene": scene.name}
    try:
        regions, ids, feats, trace, mode, clf, secs = run_isolated(
            src, scene.ctx_path, seed, scene.name, request,
            timeout=cfg.get("timeout_s", 600), n_threads=cfg.get("threads_per_job", 1))
        ev = Evaluated(regions, ids, feats, mode, clf)
        rf = ev.regions
        # 1) pixel hold-out
        val_regions = np.unique(rf[split.s_val])
        mp_ = ev.predict(split.s_in, labels, val_regions, seed, rng)
        yv = labels[split.s_val]
        pv = _pix_pred(ev, mp_, split.s_val)
        oa_val = 100.0 * np.mean(pv == yv)
        mf1 = f1_score(yv, pv, labels=np.unique(yv), average="macro", zero_division=0)
        # 2) region-grouped hold-out
        tr_regions = labeling["tr_regions"]
        if tr_regions.size >= 4:
            perm = rng.permutation(tr_regions)
            n_hold = int(np.ceil(cfg.get("group_holdout_frac", 0.25) * tr_regions.size))
            held, kept = perm[:n_hold], perm[n_hold:]
            in_held = np.isin(rf[split.s_tr], held)
            tr_pix, ho_pix = split.s_tr[~in_held], split.s_tr[in_held]
            mg = ev.predict(tr_pix, labels, held, seed, rng)
            oa_grp = 100.0 * np.mean(_pix_pred(ev, mg, ho_pix) == labels[ho_pix])
        else:
            oa_grp = 0.0
        sizes = np.bincount(rf)
        out.update(
            fitness=(oa_val + oa_grp + 100.0 * mf1) / 3.0,
            oa_val=oa_val, oa_grp=oa_grp, mf1_val=float(mf1),
            n_regions=int(sizes.size), mean_region_size=float(sizes.mean()),
            purity=purity(rf, split.s_tr, labels), seconds=time.time() - t0,
            trace=trace, error=None)
    except DesignRunError as e:
        out.update(fitness=0.0, error=str(e), seconds=time.time() - t0, trace=[])
    except Exception as e:  # harness-side failure caused by odd design output
        out.update(fitness=0.0, error=f"{type(e).__name__}: {e}", seconds=time.time() - t0, trace=[])
    return out


def evaluate_design(src: str, scenes: list[Scene], splits: dict, cfg: dict, seed: int) -> dict:
    """F(D) over all scenes, scenes run in parallel processes."""
    with ThreadPoolExecutor(max_workers=cfg.get("parallel_scenes", len(scenes))) as ex:
        futs = {s.name: ex.submit(score_search, src, s, splits[s.name], cfg, seed) for s in scenes}
        per = {k: f.result() for k, f in futs.items()}
    return {"fitness": float(np.mean([p["fitness"] for p in per.values()])), "per_scene": per}


# --------------------------------------------------------------------------
# final test evaluation (Section 8.5), used after the search
# --------------------------------------------------------------------------
def score_test(src: str, scene: Scene, s_tr: np.ndarray, s_te: np.ndarray, cfg: dict, seed: int,
               selector: str | None = None, selector_cfg: dict | None = None) -> dict:
    """Train on s_tr, score on s_te. selector='pso'|'goa' gives MRG-PSO / MRG-ASVM
    (feature selection on the training regions, region LABELING only)."""
    labels = scene.labels
    rng = np.random.default_rng(seed + 104729)
    allpix = np.union1d(s_tr, s_te)
    store = {}

    def request(regions, mode):
        rf = regions.ravel()
        target = np.unique(rf[allpix])
        store["target"] = target
        if mode == "pixel_vote":
            return pixel_ids_for_vote(rf, target, rng, cfg.get("max_vote_pixels", 200), s_tr)
        return target

    t0 = time.time()
    regions, ids, feats, trace, mode, clf, _ = run_isolated(
        src, scene.ctx_path, seed, scene.name, request,
        timeout=cfg.get("timeout_s", 600), n_threads=cfg.get("threads_per_job", 1))
    ev = Evaluated(regions, ids, feats, mode, clf)
    rf = ev.regions
    n_sel = ev.feats.shape[1]
    if selector:
        from .feature_selection import SELECTORS
        if mode != "region":
            raise ValueError("feature selection is implemented for LABELING='region' only")
        sc_cfg = dict(selector_cfg or {})
        mask_path = sc_cfg.pop("mask_path", None)
        if mask_path and os.path.exists(mask_path):
            mask = np.load(mask_path)  # subset chosen once on the search split
        else:
            rid, ry = region_majority(rf, s_tr, labels)
            mask, _ = SELECTORS[selector](ev.feats[ev.row_of[rid]], ry, seed=seed, **sc_cfg)
            if mask.sum() == 0:
                mask[:] = True
            if mask_path:
                from .feature_selection import HISTORY
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                np.save(mask_path, mask)
                np.save(mask_path.replace(".npy", "_history.npy"), np.array(HISTORY))
        ev.feats = ev.feats[:, mask]
        n_sel = int(mask.sum())
    te_regions = np.unique(rf[s_te])
    mp_ = ev.predict(s_tr, labels, te_regions, seed, rng)
    yt = labels[s_te]
    pt = _pix_pred(ev, mp_, s_te)
    classes = np.unique(yt)
    per_class = {int(c): float(np.mean(pt[yt == c] == c)) for c in classes}
    # test pixels whose region holds no training pixel (leak-free subset)
    tr_regions = np.unique(rf[s_tr])
    clean = ~np.isin(rf[s_te], tr_regions)
    return {
        "scene": scene.name,
        "oa": 100.0 * float(np.mean(pt == yt)),
        "aa": 100.0 * float(np.mean(list(per_class.values()))),
        "kappa": float(cohen_kappa_score(yt, pt)),
        "mf1": float(f1_score(yt, pt, labels=classes, average="macro", zero_division=0)),
        "per_class_acc": per_class,
        "purity": purity(rf, allpix, labels),
        "n_regions": int(rf.max() + 1),
        "n_features": int(n_sel),
        "oa_regions_without_train_pixels": (100.0 * float(np.mean(pt[clean] == yt[clean]))
                                            if clean.any() else None),
        "frac_test_in_regions_with_train_pixels": float(1 - clean.mean()),
        "seconds": time.time() - t0,
        "trace": trace,
        "pred": pt.astype(np.int16).tolist(),
        "test_idx": s_te.astype(np.int64).tolist(),
    }


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def score_test_multi(src: str, scene: Scene, splits: list, cfg: dict, seed: int,
                     selector: str | None = None, selector_cfg: dict | None = None) -> list:
    """Like score_test, for many (tag, s_tr, s_te) splits with one run of the design.
    The design only sees unlabelled data and the regions it is asked to describe (all regions
    holding any labelled pixel), so its output does not depend on the split and can be reused."""
    labels = scene.labels
    rng = np.random.default_rng(seed + 104729)
    allpix = scene.labelled

    def request(regions, mode):
        rf = regions.ravel()
        target = np.unique(rf[allpix])
        if mode == "pixel_vote":
            return pixel_ids_for_vote(rf, target, rng, cfg.get("max_vote_pixels", 200), allpix)
        return target

    t0 = time.time()
    regions, ids, feats, trace, mode, clf, _ = run_isolated(
        src, scene.ctx_path, seed, scene.name, request,
        timeout=cfg.get("timeout_s", 600), n_threads=cfg.get("threads_per_job", 1))
    t_design = time.time() - t0
    out = []
    for tag, s_tr, s_te in splits:
        t1 = time.time()
        ev = Evaluated(regions, ids, feats, mode, clf)
        rf = ev.regions
        n_sel = ev.feats.shape[1]
        if selector:
            from .feature_selection import SELECTORS
            sc_cfg = dict(selector_cfg or {})
            mask_path = sc_cfg.pop("mask_path", None)
            if mask_path and os.path.exists(mask_path):
                mask = np.load(mask_path)
            else:
                rid, ry = region_majority(rf, s_tr, labels)
                mask, _ = SELECTORS[selector](ev.feats[ev.row_of[rid]], ry, seed=seed, **sc_cfg)
            ev.feats = ev.feats[:, mask]
            n_sel = int(mask.sum())
        te_regions = np.unique(rf[s_te])
        mp_ = ev.predict(s_tr, labels, te_regions, seed, rng)
        yt = labels[s_te]
        pt = _pix_pred(ev, mp_, s_te)
        classes = np.unique(yt)
        per_class = {int(c): float(np.mean(pt[yt == c] == c)) for c in classes}
        clean = ~np.isin(rf[s_te], np.unique(rf[s_tr]))
        out.append({
            "tag": tag, "scene": scene.name,
            "oa": 100.0 * float(np.mean(pt == yt)),
            "aa": 100.0 * float(np.mean(list(per_class.values()))),
            "kappa": float(cohen_kappa_score(yt, pt)),
            "mf1": float(f1_score(yt, pt, labels=classes, average="macro", zero_division=0)),
            "per_class_acc": per_class,
            "purity": purity(rf, allpix, labels),
            "n_regions": int(rf.max() + 1), "n_features": int(n_sel),
            "oa_regions_without_train_pixels": (100.0 * float(np.mean(pt[clean] == yt[clean]))
                                                if clean.any() else None),
            "frac_test_in_regions_with_train_pixels": float(1 - clean.mean()),
            "seconds": time.time() - t1 + (t_design if not out else 0.0),
            "trace": trace, "pred": pt.astype(np.int16).tolist()})
    return out


# --------------------------------------------------------------------------
# edge-preserving filtering competitor (guided-filter variant of Kang et al. 2014)
# --------------------------------------------------------------------------
def _box(x, r):
    from scipy import ndimage
    return ndimage.uniform_filter(x, size=2 * r + 1, mode="reflect")


def guided_filter(I, p, r, eps):
    mI, mp = _box(I, r), _box(p, r)
    a = (_box(I * p, r) - mI * mp) / (_box(I * I, r) - mI * mI + eps)
    b = mp - a * mI
    return _box(a, r) * I + _box(b, r)


def score_epf_multi(scene: Scene, splits: list, seed: int, C: float = 100.0, r: int = 3,
                    eps: float = 0.01) -> list:
    """Pixel-wise RBF SVM on the ten PCs; its one-vs-rest decision maps are smoothed by a guided
    filter with the first PC (scaled to [0, 1]) as guide; each pixel takes the arg-max class.
    Label-dependent, so it runs in the harness, not as a design."""
    with np.load(scene.ctx_path) as z:
        pcs = z["pcs"].astype(np.float64)
    h, w, k = pcs.shape
    X = pcs.reshape(-1, k)
    g = pcs[..., 0]
    g = (g - g.min()) / (g.max() - g.min() + 1e-12)
    labels = scene.labels
    out = []
    for tag, s_tr, s_te in splits:
        t0 = time.time()
        sc = StandardScaler().fit(X[s_tr])
        clf = SVC(kernel="rbf", C=C, gamma="scale", decision_function_shape="ovr",
                  random_state=seed).fit(sc.transform(X[s_tr]), labels[s_tr])
        # The guided filter is two nested box filters of radius r, so a test pixel's filtered
        # value depends only on decision values within 2r pixels; compute those only.
        from scipy import ndimage
        need = np.zeros(h * w, bool)
        need[s_te] = True
        need = ndimage.binary_dilation(need.reshape(h, w), np.ones((4 * r + 1, 4 * r + 1), bool)).ravel()
        idx = np.flatnonzero(need)
        Xs = sc.transform(X[idx])
        Dn = np.vstack([clf.decision_function(Xs[i:i + 50000]) for i in range(0, Xs.shape[0], 50000)])
        if Dn.ndim == 1:
            Dn = np.stack([-Dn, Dn], 1)
        D = np.zeros((h * w, Dn.shape[1]))
        D[idx] = Dn
        F = np.stack([guided_filter(g, D[:, j].reshape(h, w), r, eps) for j in range(D.shape[1])], -1)
        pt = clf.classes_[np.argmax(F.reshape(-1, D.shape[1])[s_te], 1)]
        yt = labels[s_te]
        classes = np.unique(yt)
        per_class = {int(c): float(np.mean(pt[yt == c] == c)) for c in classes}
        out.append({"tag": tag, "scene": scene.name, "oa": 100.0 * float(np.mean(pt == yt)),
                    "aa": 100.0 * float(np.mean(list(per_class.values()))),
                    "kappa": float(cohen_kappa_score(yt, pt)),
                    "mf1": float(f1_score(yt, pt, labels=classes, average="macro", zero_division=0)),
                    "per_class_acc": per_class, "purity": None, "n_regions": int(h * w),
                    "n_features": k, "oa_regions_without_train_pixels": 100.0 * float(np.mean(pt == yt)),
                    "frac_test_in_regions_with_train_pixels": 0.0, "seconds": time.time() - t0,
                    "trace": [], "pred": pt.astype(np.int16).tolist()})
    return out
