"""Final evaluation on the held-out 30% test split (Section 8.5).

Methods: Baseline MRG, MRG-PSO, MRG-ASVM, DoE (optional), LLM-MRG, LLM-RB.
Protocols: multi-split stratified 70/30 (final.split_seeds) and a spatially
disjoint tile split (final.spatial_seeds). Outputs go to <run_dir>/final/.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import _path  # noqa: F401
from llm_mrg.config import load_config, prepare_scenes
from llm_mrg.doe import run_doe
from llm_mrg.harness import save_json, score_test, spatial_block_split, stratified_split
from llm_mrg.sandbox import DesignRunError
from llm_mrg.stats import friedman, mcnemar, wilcoxon_pairs


def _fs(fs, sc, selector):
    """fs.mask_dir set: one subset per scene and selector, chosen on the search split."""
    if not selector or not fs.get("mask_dir"):
        return {k: v for k, v in fs.items() if k != "mask_dir"}
    d = {k: v for k, v in fs.items() if k != "mask_dir"}
    d["mask_path"] = os.path.join(fs["mask_dir"], f"{sc.name}_{selector}.npy")
    return d


def _job(src, sc, s_tr, s_te, hcfg, seed, selector, fs):
    import time
    t0 = time.time()
    r = score_test(src, sc, s_tr, s_te, hcfg, seed, selector, fs)
    r["seconds"] = time.time() - t0
    return r


def methods(cfg, scenes):
    sel = json.load(open(os.path.join(cfg["run_dir"], "selection.json")))
    d0 = open(cfg["seed_design"]).read()
    fs = cfg["final"]["feature_selection"]
    m = {"Baseline MRG": (d0, None), "MRG-PSO": (d0, "pso"), "MRG-ASVM": (d0, "goa")}
    if cfg["final"].get("doe"):
        n_search = len([f for f in os.listdir(os.path.join(cfg["run_dir"], "designs")) if f.endswith(".py")])
        budget = cfg["final"].get("doe_budget") or max(1, n_search - 1)
        done = os.path.join(cfg["run_dir"], "doe", "doe_results.json")
        best = (json.load(open(done))["best"] if os.path.exists(done)
                else run_doe(cfg, scenes, os.path.join(cfg["run_dir"], "doe"), budget))
        m["DoE"] = (open(best["path"]).read(), None)
    for k in ("LLM-MRG", "LLM-RB"):
        if sel.get(k):
            m[k] = (open(sel[k]["path"]).read(), None)
    return m, fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    scenes = prepare_scenes(cfg)
    M, fs = methods(cfg, scenes)
    out_dir = os.path.join(cfg["run_dir"], "final")
    os.makedirs(out_dir, exist_ok=True)
    hcfg = dict(cfg["harness"])
    results = []  # flat records
    preds = {}    # (protocol, seed, scene, method) -> (y, pred)

    fs_seeds = cfg["final"].get("fs_split_seeds", cfg["final"]["split_seeds"])
    jobs = []
    for seed in cfg["final"]["split_seeds"]:
        for sc in scenes:
            s_tr, s_te = stratified_split(sc, cfg["train_frac"], seed)
            for name, (src, selector) in M.items():
                if selector and seed not in fs_seeds:
                    continue
                jobs.append(("random", seed, sc, s_tr, s_te, name, src, selector))
    for seed in cfg["final"].get("spatial_seeds", []):
        for sc in scenes:
            s_tr, s_te = spatial_block_split(sc, cfg["train_frac"], seed, cfg["final"]["spatial_tile"])
            for name, (src, selector) in M.items():
                if selector:
                    continue
                jobs.append(("spatial", seed, sc, s_tr, s_te, name, src, selector))
    # slow jobs first so the two workers stay busy
    jobs.sort(key=lambda j: (j[7] is None, j[2].name != "houston"))

    with ProcessPoolExecutor(max_workers=cfg["final"].get("workers", 2),
                             mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_job, src, sc, s_tr, s_te, hcfg, seed, selector, _fs(fs, sc, selector)): (p, seed, sc, s_te, name)
                for p, seed, sc, s_tr, s_te, name, src, selector in jobs}
        for f in as_completed(futs):
            protocol, seed, sc, s_te, name = futs[f]
            try:
                r = f.result()
            except Exception as e:
                print(f"  {protocol} s{seed} {sc.name} {name}: FAILED {e}", flush=True)
                continue
            preds[(protocol, seed, sc.name, name)] = (sc.labels[s_te], np.array(r.pop("pred")))
            r.pop("test_idx")
            r.update(protocol=protocol, split_seed=seed, method=name)
            results.append(r)
            print(f"  {protocol} s{seed} {sc.name:18s} {name:13s} OA={r['oa']:.2f} AA={r['aa']:.2f} "
                  f"k={r['kappa']:.4f} OA_noshare={r['oa_regions_without_train_pixels']} "
                  f"t={r['seconds']:.0f}s", flush=True)
    save_json(os.path.join(out_dir, "results.json"), results)

    # summary table: mean and sd over split seeds
    agg = defaultdict(list)
    for r in results:
        agg[(r["protocol"], r["scene"], r["method"])].append(r)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "scene", "method", "n_splits", "OA_mean", "OA_sd", "AA_mean", "AA_sd",
                    "kappa_mean", "mF1_mean", "purity_mean", "OA_no_shared_region_mean", "n_regions_mean"])
        for (p, s, m), rs in sorted(agg.items()):
            g = lambda k: np.array([x[k] for x in rs if x[k] is not None], float)
            ns = g("oa_regions_without_train_pixels")
            w.writerow([p, s, m, len(rs), g("oa").mean(), g("oa").std(ddof=1) if len(rs) > 1 else 0,
                        g("aa").mean(), g("aa").std(ddof=1) if len(rs) > 1 else 0, g("kappa").mean(),
                        g("mf1").mean(), g("purity").mean(), ns.mean() if ns.size else "", g("n_regions").mean()])

    # tests: LLM-MRG (or LLM-RB) against every other method
    tests = {"mcnemar": [], "wilcoxon_oa": {}, "friedman_oa": None}
    ref = "LLM-MRG" if "LLM-MRG" in M else "LLM-RB"
    for (p, sd, s, m), (y, pr) in preds.items():
        if m != ref and (p, sd, s, ref) in preds:
            tests["mcnemar"].append({"protocol": p, "seed": sd, "scene": s, "a": ref, "b": m,
                                     **mcnemar(y, preds[(p, sd, s, ref)][1], pr)})
    blocks = sorted({(r["scene"], r["split_seed"]) for r in results if r["protocol"] == "random"})
    table = {m: [next((r["oa"] for r in results if r["protocol"] == "random" and r["method"] == m
                       and (r["scene"], r["split_seed"]) == b), np.nan) for b in blocks] for m in M}
    for m in M:
        if m != ref and not np.any(np.isnan(table[m] + table[ref])):
            tests["wilcoxon_oa"][f"{ref} vs {m}"] = wilcoxon_pairs(table[ref], table[m])
    cols = [table[m] for m in M if not np.any(np.isnan(table[m]))]
    tests["friedman_oa"] = friedman(*cols)
    save_json(os.path.join(out_dir, "tests.json"), tests)
    print(f"wrote {out_dir}/summary.csv, results.json, tests.json")


if __name__ == "__main__":
    main()
