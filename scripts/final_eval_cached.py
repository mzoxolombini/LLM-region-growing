"""Final evaluation with one design run per (method, scene), reused across all splits.
Each (method, scene) job is saved to <run_dir>/final/jobs/ so the script can be re-run to add
methods (e.g. DoE once it has finished). --summarise builds results.json / summary.csv / tests.json.
"""
import argparse, csv, json, os, glob
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np


def _load(f):
    """Job files are stored gzipped in the repository (*.json.gz)."""
    import gzip, json
    return json.load(gzip.open(f, "rt")) if f.endswith(".gz") else json.load(open(f))

import _path  # noqa: F401
from llm_mrg.config import load_config, prepare_scenes
from llm_mrg.harness import save_json, score_test_multi, spatial_block_split, stratified_split
from llm_mrg.stats import friedman, mcnemar, wilcoxon_pairs


def methods(cfg):
    sel = json.load(open(os.path.join(cfg["run_dir"], "selection.json")))
    d0 = open(cfg["seed_design"]).read()
    m = {"Baseline MRG": (d0, None), "MRG-PSO": (d0, "pso"), "MRG-ASVM": (d0, "goa")}
    done = os.path.join(cfg["run_dir"], "doe", "best.json")
    if os.path.exists(done):
        m["DoE"] = (open(json.load(open(done))["path"]).read(), None)
    for k in ("LLM-MRG", "LLM-RB"):
        if sel.get(k):
            m[k] = (open(sel[k]["path"]).read(), None)
    return m


def job(cfg, sc, name, src, selector, out_path):
    f = cfg["final"]
    splits = [(f"random_{s}", *stratified_split(sc, cfg["train_frac"], s)) for s in f["split_seeds"]]
    if not selector:
        splits += [(f"spatial_{s}", *spatial_block_split(sc, cfg["train_frac"], s, f["spatial_tile"]))
                   for s in f.get("spatial_seeds", [])]
    fs = dict(f["feature_selection"])
    if selector:
        fs["mask_path"] = os.path.join(fs.pop("mask_dir"), f"{sc.name}_{selector}.npy")
    else:
        fs.pop("mask_dir", None)
    res = score_test_multi(src, sc, splits, cfg["harness"], cfg["seed"], selector, fs)
    for r, (tag, _, s_te) in zip(res, splits):
        r["method"] = name
        r["test_idx_hash"] = int(np.sum(s_te) % 1000003)
    save_json(out_path, res)
    return f"{name} {sc.name}: " + " ".join(f"{r['tag']}={r['oa']:.2f}" for r in res)


def summarise(cfg, scenes):
    out = os.path.join(cfg["run_dir"], "final")
    R = []
    for f in sorted(glob.glob(f"{out}/jobs/*.json*")):
        R += _load(f)
    preds = {}
    labels = {sc.name: sc for sc in scenes}
    for r in R:
        r["protocol"], r["split_seed"] = r["tag"].split("_")[0], int(r["tag"].split("_")[1])
        preds[(r["protocol"], r["split_seed"], r["scene"], r["method"])] = np.array(r.pop("pred"))
    save_json(f"{out}/results.json", R)
    agg = defaultdict(list)
    for r in R:
        agg[(r["protocol"], r["scene"], r["method"])].append(r)
    with open(f"{out}/summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protocol", "scene", "method", "n", "OA", "OA_sd", "AA", "AA_sd", "kappa", "mF1",
                    "OA_no_shared", "frac_shared", "n_regions"])
        for (p, s, m), rs in sorted(agg.items()):
            g = lambda k: np.array([x[k] for x in rs if x[k] is not None], float)
            sd = lambda k: g(k).std(ddof=1) if len(g(k)) > 1 else 0.0
            ns = g("oa_regions_without_train_pixels")
            w.writerow([p, s, m, len(rs), g("oa").mean(), sd("oa"), g("aa").mean(), sd("aa"),
                        g("kappa").mean(), g("mf1").mean(), ns.mean() if ns.size else "",
                        g("frac_test_in_regions_with_train_pixels").mean(), g("n_regions").mean()])
    # tests against LLM-MRG and LLM-RB
    tests = {"mcnemar": [], "wilcoxon_oa": {}, "friedman_oa": None}
    M = sorted({r["method"] for r in R})
    for ref in ("LLM-MRG", "LLM-RB"):
        for (p, sd_, s, m), pr in preds.items():
            if m != ref and (p, sd_, s, ref) in preds:
                sc = labels[s]
                _, s_te = (stratified_split(sc, cfg["train_frac"], sd_) if p == "random" else
                           spatial_block_split(sc, cfg["train_frac"], sd_, cfg["final"]["spatial_tile"]))
                tests["mcnemar"].append({"protocol": p, "seed": sd_, "scene": s, "a": ref, "b": m,
                                         **mcnemar(sc.labels[s_te], preds[(p, sd_, s, ref)], pr)})
    blocks = sorted({(r["scene"], r["split_seed"]) for r in R if r["protocol"] == "random"})
    table = {m: [next((r["oa"] for r in R if r["protocol"] == "random" and r["method"] == m
                       and (r["scene"], r["split_seed"]) == b), np.nan) for b in blocks] for m in M}
    for ref in ("LLM-MRG", "LLM-RB"):
        for m in M:
            if m != ref and ref in table and not np.any(np.isnan(np.array(table[m]) + np.array(table[ref]))):
                tests["wilcoxon_oa"][f"{ref} vs {m}"] = wilcoxon_pairs(table[ref], table[m])
    tests["friedman_oa"] = friedman(*[table[m] for m in M])
    tests["friedman_methods"] = M
    save_json(f"{out}/tests.json", tests)
    print(f"summarised {len(R)} records")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*", help="evaluate only these methods")
    a = ap.parse_args()
    cfg = load_config(a.config)
    scenes = prepare_scenes(cfg)
    if a.summarise:
        summarise(cfg, scenes)
    else:
        M = methods(cfg)
        jdir = os.path.join(cfg["run_dir"], "final", "jobs")
        os.makedirs(jdir, exist_ok=True)
        todo = [(sc, n, src, sel, os.path.join(jdir, f"{n.replace(' ', '_')}__{sc.name}.json"))
                for n, (src, sel) in M.items() for sc in scenes]
        todo = [t for t in todo if not os.path.exists(t[4]) and (not a.only or t[1] in a.only)]
        todo.sort(key=lambda t: (t[1] != "LLM-RB", t[0].name not in ("salinas", "pavia_university")))
        with ProcessPoolExecutor(cfg["final"].get("workers", 2), mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(job, cfg, *t) for t in todo]
            for f in as_completed(futs):
                try:
                    print(f.result(), flush=True)
                except Exception as e:
                    print("FAILED", repr(e)[:300], flush=True)
