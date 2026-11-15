"""Work that does not depend on the LLM search, run alongside it:
  --doe : the DoE competitor (Latin hypercube, budget = final.doe_budget)
  --fs  : PSO / GOA feature subsets for every scene, chosen on the search split
"""
import argparse, os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import _path  # noqa: F401
from llm_mrg.config import load_config, prepare_scenes
from llm_mrg.doe import run_doe
from llm_mrg.harness import score_test, stratified_split


def fs_job(cfg, sc, sel):
    fs = dict(cfg["final"]["feature_selection"])
    fs["mask_path"] = os.path.join(fs.pop("mask_dir"), f"{sc.name}_{sel}.npy")
    s_tr, s_te = stratified_split(sc, cfg["train_frac"], cfg["seed"])
    r = score_test(open(cfg["seed_design"]).read(), sc, s_tr, s_te, cfg["harness"], cfg["seed"], sel, fs)
    return f"{sc.name} {sel}: {r['n_features']} features, OA={r['oa']:.2f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--doe", action="store_true")
    ap.add_argument("--fs", action="store_true")
    ap.add_argument("--doe_range", nargs=2, type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    scenes = prepare_scenes(cfg)
    if a.fs:
        with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
            for msg in ex.map(fs_job, [cfg] * 10, [s for s in scenes for _ in (0, 1)], ["pso", "goa"] * 5):
                print(msg, flush=True)
    if a.doe:
        cfg["harness"]["parallel_scenes"] = 1
        rng = a.doe_range or (0, None)
        run_doe(cfg, scenes, os.path.join(cfg["run_dir"], "doe"), cfg["final"]["doe_budget"], first=rng[0], last=rng[1])
