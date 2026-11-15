"""End-to-end smoke test on two small synthetic scenes with the mock LLM.
No data download, no API key. Checks that every stage runs, not that results mean anything.

  python scripts/smoke_test.py
"""
import shutil
import subprocess
import sys

import _path  # noqa: F401

RUN = "runs/smoke"
SET = ["scenes=[synthetic_0,synthetic_1]", "cache_dir=cache_smoke", f"run_dir={RUN}",
       "llm.provider=mock", "llm.model=mock", "search.mu=3", "search.lambda=4", "search.generations=2",
       "harness.timeout_s=300", "final.split_seeds=[0,1]", "final.spatial_tile=10",
       "final.feature_selection={n_particles: 4, iters: 3, alpha: 0.99, folds: 3}",
       "final.doe_budget=3"]

if __name__ == "__main__":
    shutil.rmtree(RUN, ignore_errors=True)
    for script in ("scripts/run_search.py", "scripts/run_final_eval.py"):
        print(f"=== {script}")
        r = subprocess.run([sys.executable, script, "--config", "config.yaml", "--set", *SET])
        if r.returncode:
            sys.exit(r.returncode)
    print("smoke test passed")
