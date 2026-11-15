"""Elitist (mu + lambda) evolutionary loop with the LLM as the only variation
operator (Algorithm 6, Section 8.4.3).

Run directory layout
  runs/<name>/config.json
  runs/<name>/llm_log.jsonl            every prompt and reply
  runs/<name>/designs/<id>.py          every design's code
  runs/<name>/designs/<id>.json        its fitness, per-scene feedback, lineage
  runs/<name>/population_gen<g>.json   ids of the survivors after generation g
The loop resumes from the last completed generation if the directory exists.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import time

import numpy as np

from .harness import evaluate_design, make_search_split, save_json
from .llm_client import LLM, NeedReply, extract_code
from .prompts import SYSTEM, build_prompt, task_section
from .sandbox import load_design

CYCLE = ("E1", "E2", "M1", "M2")
N_PARENTS = {"E1": 0, "E2": 2, "M1": 1, "M2": 1}


def _description(src: str) -> str:
    try:
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DESCRIPTION" for t in node.targets):
                return str(ast.literal_eval(node.value))
    except Exception:
        pass
    return ""


def _strip_params(src: str) -> str:
    """AST dump of the module with PARAMS/CLASSIFIER values blanked (for the M2 check)."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in ("PARAMS", "CLASSIFIER")
                                                for t in node.targets):
            node.value = ast.Constant(None)
    return ast.dump(tree, annotate_fields=False)


def m2_violation(parent_src: str, child_src: str) -> str | None:
    try:
        if _strip_params(parent_src) != _strip_params(child_src):
            return "M2 may change only PARAMS and CLASSIFIER, but other code was changed"
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    return None


def rank_select(pop: list[dict], k: int, rng) -> list[dict]:
    """k parents without replacement, probability proportional to rank (best = n)."""
    order = sorted(pop, key=lambda e: -e["fitness"])
    n = len(order)
    w = np.arange(n, 0, -1, dtype=float)
    idx = rng.choice(n, size=k, replace=False, p=w / w.sum())
    return [order[i] for i in idx]


class Search:
    def __init__(self, cfg: dict, scenes: list, run_dir: str):
        self.cfg = cfg
        self.scenes = scenes
        self.run_dir = run_dir
        os.makedirs(os.path.join(run_dir, "designs"), exist_ok=True)
        save_json(os.path.join(run_dir, "config.json"), cfg)
        s = cfg["search"]
        self.mu, self.lam, self.G = s["mu"], s["lambda"], s["generations"]
        self.seed = cfg["seed"]
        self.rng = np.random.default_rng(self.seed)
        self.llm = LLM(cfg["llm"], os.path.join(run_dir, "llm_log.jsonl"))
        self.splits = {sc.name: make_search_split(sc, cfg["train_frac"], cfg["inner_frac"], self.seed)
                       for sc in scenes}
        self.seen: dict[str, dict] = {}
        info = [{"name": sc.name, "H": sc.gt.shape[0], "W": sc.gt.shape[1],
                 "B": int(np.load(sc.ctx_path)["bands"].shape[2]),
                 "K": int(sc.gt.max()), "n_labelled": int(sc.labelled.size)} for sc in scenes]
        self.seed_src = open(cfg["seed_design"]).read()
        self.task = task_section(info, self.seed_src, cfg["harness"]["timeout_s"])

    # ------------------------------------------------------------------
    def _evaluate(self, src: str, gen: int, op: str, parents: list[str], k: int,
                  pre_error: str | None = None) -> dict:
        h = hashlib.sha1(src.encode()).hexdigest()[:10]
        did = f"g{gen:02d}_{k:02d}_{op}_{h}"
        if pre_error is None and h in self.seen:  # identical code: reuse the score
            old = self.seen[h]
            entry = dict(old, id=did, gen=gen, op=op, parents=parents, duplicate_of=old["id"])
        else:
            t0 = time.time()
            if pre_error:
                res = {"fitness": 0.0, "per_scene": {sc.name: {"fitness": 0.0, "error": pre_error}
                                                     for sc in self.scenes}}
            else:
                try:
                    load_design(src)  # syntax and import errors surface here quickly
                    res = evaluate_design(src, self.scenes, self.splits, self.cfg["harness"], self.seed)
                except Exception as e:
                    res = {"fitness": 0.0, "per_scene": {sc.name: {"fitness": 0.0, "error": f"{type(e).__name__}: {e}"}
                                                         for sc in self.scenes}}
            entry = {"id": did, "gen": gen, "op": op, "parents": parents, "src": src,
                     "description": _description(src), "fitness": res["fitness"],
                     "per_scene": res["per_scene"], "eval_seconds": time.time() - t0}
            if pre_error is None:
                self.seen[h] = entry
        with open(os.path.join(self.run_dir, "designs", f"{did}.py"), "w") as f:
            f.write(src)
        save_json(os.path.join(self.run_dir, "designs", f"{did}.json"),
                  {k_: v for k_, v in entry.items() if k_ != "src"})
        print(f"  [{did}] F={entry['fitness']:.2f}  {entry['description'][:90]}", flush=True)
        return entry

    def _offspring(self, gen: int, k: int, pop: list[dict]) -> dict:
        op = CYCLE[(k - 1) % len(CYCLE)]
        need = N_PARENTS[op]
        if need > len(pop):
            # ASSUMPTION: the chapter does not say what E2 does while the population
            # holds a single design (generation 1). We fall back to M1.
            op, need = "M1", 1
        parents = rank_select(pop, need, self.rng) if need else []
        prompt = build_prompt(op, self.task, parents, pop)
        meta = {"gen": gen, "k": k, "op": op, "parents": [p["id"] for p in parents],
                "fallback_code": (parents[0] if parents else max(pop, key=lambda e: e["fitness"]))["src"]}
        try:
            reply = self.llm.complete(SYSTEM, prompt, meta)
        except NeedReply:
            return None
        except RuntimeError as e:
            return self._evaluate("# LLM call failed\n", gen, op, meta["parents"], k, pre_error=str(e))
        src = extract_code(reply)
        if not src:
            return self._evaluate("# no code in reply\n", gen, op, meta["parents"], k,
                                  pre_error="the reply contained no Python code block")
        err = m2_violation(parents[0]["src"], src) if op == "M2" else None
        return self._evaluate(src, gen, op, meta["parents"], k, pre_error=err)

    # ------------------------------------------------------------------
    def _load_state(self):
        gens = sorted(int(f[14:-5]) for f in os.listdir(self.run_dir)
                      if f.startswith("population_gen") and f.endswith(".json"))
        if not gens:
            return 0, None
        g = gens[-1]
        ids = json.load(open(os.path.join(self.run_dir, f"population_gen{g}.json")))["ids"]
        pop = []
        ddir = os.path.join(self.run_dir, "designs")
        for f in sorted(os.listdir(ddir)):
            if f.endswith(".json"):
                e = json.load(open(os.path.join(ddir, f)))
                e["src"] = open(os.path.join(ddir, f[:-5] + ".py")).read()
                h = e["id"].rsplit("_", 1)[-1]
                if not any((r.get("error") or "").startswith(("the reply", "M2 ", "LLM failed"))
                           for r in e["per_scene"].values()):
                    self.seen.setdefault(h, e)
                if e["id"] in ids:
                    pop.append(e)
        return g, pop

    def run(self) -> dict:
        g0, pop = self._load_state()
        if pop is None:
            print("generation 0: seed design", flush=True)
            pop = [self._evaluate(self.seed_src, 0, "D0", [], 0)]
            save_json(os.path.join(self.run_dir, "population_gen0.json"), {"ids": [e["id"] for e in pop]})
        for g in range(g0 + 1, self.G + 1):
            print(f"generation {g}", flush=True)
            self.rng = np.random.default_rng([self.seed, g])  # same parents on a re-run
            off = [self._offspring(g, k, pop) for k in range(1, self.lam + 1)]
            if any(o is None for o in off):
                print(f"waiting for replies in {self.llm.reply_dir} (generation {g})")
                return None
            pop = sorted(pop + off, key=lambda e: -e["fitness"])[: self.mu]
            save_json(os.path.join(self.run_dir, f"population_gen{g}.json"),
                      {"ids": [e["id"] for e in pop], "fitness": [e["fitness"] for e in pop]})
        best = max(pop, key=lambda e: e["fitness"])
        print(f"best: {best['id']} F={best['fitness']:.2f}")
        return best
