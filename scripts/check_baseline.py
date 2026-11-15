"""Consistency check of Section 8.5.1: the seed design run through the harness
(isolated process) must give exactly the same test OA as the stand-alone
baseline. The thesis reports 89.82% on Indian Pines; this script also prints
how far the reconstruction here is from that number.
"""
import argparse

import numpy as np

import _path  # noqa: F401
from llm_mrg.config import load_config
from llm_mrg.context import load_context
from llm_mrg.harness import (Evaluated, _pix_pred, prepare_scene, score_test, stratified_split)
from llm_mrg.sandbox import load_design

TARGET = {"indian_pines": 89.82}


def standalone(src, sc, s_tr, s_te, seed):
    mod = load_design(src)
    ctx = load_context(sc.ctx_path, seed, sc.name)
    from llm_mrg.blocks import relabel
    reg = relabel(np.asarray(mod.segment(ctx, dict(mod.PARAMS))))
    rf = reg.ravel()
    target = np.unique(rf[np.union1d(s_tr, s_te)])
    F = mod.region_features(ctx, reg, target, dict(mod.PARAMS))
    ev = Evaluated(reg, target, np.nan_to_num(np.asarray(F, float)), mod.LABELING, mod.CLASSIFIER)
    mp_ = ev.predict(s_tr, sc.labels, np.unique(rf[s_te]), seed, np.random.default_rng(0))
    return 100 * np.mean(_pix_pred(ev, mp_, s_te) == sc.labels[s_te])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--scene", default="indian_pines")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    sc = prepare_scene(a.scene, cfg["data_dir"], cfg["cache_dir"], seed=cfg["seed"])
    src = open(cfg["seed_design"]).read()
    s_tr, s_te = stratified_split(sc, cfg["train_frac"], cfg["seed"])
    h = score_test(src, sc, s_tr, s_te, cfg["harness"], cfg["seed"])["oa"]
    s = standalone(src, sc, s_tr, s_te, cfg["seed"])
    print(f"harness OA = {h:.2f}  stand-alone OA = {s:.2f}  identical: {abs(h - s) < 1e-9}")
    if a.scene in TARGET:
        print(f"thesis value = {TARGET[a.scene]:.2f}  difference = {h - TARGET[a.scene]:+.2f}")
