"""Run the LLM automated design search (Algorithm 6) and pick LLM-MRG / LLM-RB.

  python scripts/run_search.py --config config.yaml
  python scripts/run_search.py --config config.yaml --set llm.provider=mock search.generations=2
"""
import argparse
import os

import _path  # noqa: F401
from llm_mrg.config import load_config, prepare_scenes
from llm_mrg.evolve import Search
from llm_mrg.harness import save_json
from llm_mrg.select import choose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. search.mu=6")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    scenes = prepare_scenes(cfg)
    if Search(cfg, scenes, cfg["run_dir"]).run() is None:
        return  # manual provider: prompts written, waiting for replies
    sel = choose(cfg["run_dir"], cfg["select"]["strict_growth_rule"])
    out = {k: ({"id": v["id"], "fitness": v["fitness"], "path": v["path"],
                "description": v.get("description", "")} if isinstance(v, dict) else v)
           for k, v in sel.items()}
    save_json(os.path.join(cfg["run_dir"], "selection.json"), out)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
