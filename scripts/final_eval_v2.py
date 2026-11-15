"""Revision experiments: new protocols (buffered tiles, low-label) and new competitors
(pixel-only LLM-MRG / LLM-RB features, EPF, PSO/GOA with a larger budget).

  --group main : Baseline, DoE, LLM-MRG, LLM-RB on the new protocols; PIX-MRG, PIX-RB, EPF on all
  --group fs   : MRG-PSO / MRG-ASVM (masks from config final.feature_selection.mask_dir) on all
"""
import argparse, json, os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import _path  # noqa: F401
from llm_mrg.config import load_config, prepare_scenes
from llm_mrg.harness import (save_json, score_test_multi, score_epf_multi, stratified_split,
                             spatial_block_split, buffered_spatial_split, low_label_split)

PROT = {"random": list(range(10)), "spatial": [0, 1, 2], "buffered": [0, 1, 2, 3, 4], "low": list(range(10)),
        "tiles64": [0, 1, 2, 3, 4]}


def splits_for(cfg, sc, prots):
    out = []
    for p in prots:
        for s in PROT[p]:
            if p == "random":
                a, b = stratified_split(sc, cfg["train_frac"], s)
            elif p == "spatial":
                a, b = spatial_block_split(sc, cfg["train_frac"], s, 16)
            elif p == "buffered":
                a, b = buffered_spatial_split(sc, 0.5, s, 64, 20)
            elif p == "tiles64":  # same tiles as "buffered", no buffer (control)
                a, b = buffered_spatial_split(sc, 0.5, s, 64, -1)
            else:
                a, b = low_label_split(sc, 50, s)
            out.append((f"{p}_{s}", a, b))
    return out


def job(cfg, sc, name, src, selector, prots, out_path):
    sp = splits_for(cfg, sc, prots)
    if name == "EPF":
        res = score_epf_multi(sc, sp, cfg["seed"])
    else:
        fs = dict(cfg["final"]["feature_selection"])
        if selector:
            fs["mask_path"] = os.path.join(fs.pop("mask_dir"), f"{sc.name}_{selector}.npy")
        else:
            fs.pop("mask_dir", None)
        res = score_test_multi(src, sc, sp, cfg["harness"], cfg["seed"], selector, fs)
    for r in res:
        r["method"] = name
    save_json(out_path, res)
    return f"{name} {sc.name}: " + " ".join(f"{r['tag']}={r['oa']:.1f}" for r in res[:3]) + " ..."


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_rev.yaml")
    ap.add_argument("--group", choices=["main", "fs", "rev2"], required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config(a.config)
    scenes = prepare_scenes(cfg)
    sel = json.load(open(os.path.join(cfg["run_dir"], "selection.json")))
    d0 = open(cfg["seed_design"]).read()
    ALL = ["random", "spatial", "buffered", "low"]
    if a.group == "main":
        M = {"Baseline MRG": (d0, None, ["buffered", "low"]),
             "DoE": (open(json.load(open(os.path.join(cfg["run_dir"], "doe", "best.json")))["path"]).read(), None, ["buffered", "low"]),
             "LLM-MRG": (open(sel["LLM-MRG"]["path"]).read(), None, ["buffered", "low"]),
             "LLM-RB": (open(sel["LLM-RB"]["path"]).read(), None, ["buffered", "low"]),
             "PIX-MRG": (open("designs/pix_mrg_features.py").read(), None, ALL),
             "PIX-RB": (open("designs/pix_rb_features.py").read(), None, ALL),
             "EPF": ("", None, ALL)}
    elif a.group == "fs":
        M = {"MRG-PSO": (d0, "pso", ALL), "MRG-ASVM": (d0, "goa", ALL)}
    else:
        T = ["tiles64"]
        M = {"Baseline MRG": (d0, None, T), "MRG-PSO": (d0, "pso", T), "MRG-ASVM": (d0, "goa", T),
             "DoE": (open(json.load(open(os.path.join(cfg["run_dir"], "doe", "best.json")))["path"]).read(), None, T),
             "LLM-MRG": (open(sel["LLM-MRG"]["path"]).read(), None, T),
             "LLM-RB": (open(sel["LLM-RB"]["path"]).read(), None, T),
             "PIX-MRG": (open("designs/pix_mrg_features.py").read(), None, T),
             "PIX-RB": (open("designs/pix_rb_features.py").read(), None, T),
             "EPF": ("", None, T),
             "SVM-PC": (open("designs/svm_pc.py").read(), None, ALL + T)}
    jdir = os.path.join(cfg["run_dir"], "final", "jobs_v2" if a.group != "rev2" else "jobs_v3")
    os.makedirs(jdir, exist_ok=True)
    todo = [(sc, n, src, selr, prots, os.path.join(jdir, f"{n.replace(' ', '_')}__{sc.name}.json"))
            for n, (src, selr, prots) in M.items() for sc in scenes]
    todo = [t for t in todo if not os.path.exists(t[-1])]
    todo.sort(key=lambda t: (t[1] not in ("PIX-RB", "LLM-RB", "EPF"), t[0].name != "houston"))
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(job, cfg, *t) for t in todo]
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print("FAILED", repr(e)[:400], flush=True)
