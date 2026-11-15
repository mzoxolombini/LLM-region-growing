"""Download the four EHU scenes. Houston 2013 must be placed in data/ by hand."""
import argparse
import _path  # noqa: F401
from llm_mrg.datasets import SCENES, download

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    a = ap.parse_args()
    for s in a.scenes:
        try:
            download(s, a.data_dir)
        except FileNotFoundError as e:
            print(f"[{s}] {e}")
