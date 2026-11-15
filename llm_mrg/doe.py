"""Design-of-experiments competitor (Section 5.2.3 is not in the chapter
supplied; this is one reasonable reading). A Latin hypercube over the numeric
PARAMS and CLASSIFIER of the seed design, scored with the same search fitness
and the same evaluation budget as the LLM search. It answers "is the LLM doing
more than tuning numbers?", the same question M2 alone would answer.
"""
from __future__ import annotations

import ast
import os

import numpy as np
from scipy.stats import qmc

from .harness import evaluate_design, make_search_split, save_json

# name: (low, high, log-scale, integer)
DEFAULT_SPACE = {
    "PARAMS.t_intensity": (0.005, 0.3, True, False),
    "PARAMS.t_orientation": (np.pi / 16, np.pi, False, False),
    "PARAMS.n_bins": (16, 256, True, True),
    "PARAMS.glcm_levels": (8, 32, True, True),
    "CLASSIFIER.C": (1e-2, 1e3, True, False),
}


def substitute(src: str, values: dict) -> str:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in ("PARAMS", "CLASSIFIER"):
                d = ast.literal_eval(node.value) if not isinstance(node.value, ast.Call) else None
                if d is None:
                    continue
                for k, v in values.items():
                    top, key = k.split(".", 1)
                    if top == name:
                        d[key] = v
                node.value = ast.parse(repr(d), mode="eval").body
    return ast.unparse(tree) + "\n"


def run_doe(cfg: dict, scenes: list, run_dir: str, budget: int, space: dict | None = None,
            first: int = 0, last: int | None = None) -> dict:
    space = space or DEFAULT_SPACE
    os.makedirs(run_dir, exist_ok=True)
    seed = cfg["seed"]
    splits = {s.name: make_search_split(s, cfg["train_frac"], cfg["inner_frac"], seed) for s in scenes}
    src0 = open(cfg["seed_design"]).read()
    u = qmc.LatinHypercube(d=len(space), seed=seed).random(budget)
    best = None
    rows = []
    for i, row in enumerate(u):
        if i < first or (last is not None and i > last):
            continue
        vals = {}
        for (k, (lo, hi, lg, it)), x in zip(space.items(), row):
            v = np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))) if lg else lo + x * (hi - lo)
            vals[k] = int(round(v)) if it else float(v)
        src = substitute(src0, vals)
        res = evaluate_design(src, scenes, splits, cfg["harness"], seed)
        rows.append({"i": i, "values": vals, "fitness": res["fitness"],
                     "per_scene": {k: {kk: vv for kk, vv in v.items() if kk != "trace"}
                                   for k, v in res["per_scene"].items()}})
        with open(os.path.join(run_dir, f"doe_{i:03d}.py"), "w") as f:
            f.write(src)
        print(f"  DoE {i}: F={res['fitness']:.2f} {vals}", flush=True)
        if best is None or res["fitness"] > best["fitness"]:
            best = {"i": i, "fitness": res["fitness"], "path": os.path.join(run_dir, f"doe_{i:03d}.py")}
    save_json(os.path.join(run_dir, "doe_results.json" if (first == 0 and last is None) else f"doe_results_{first}_{last}.json"), {"rows": rows, "best": best})
    return best
