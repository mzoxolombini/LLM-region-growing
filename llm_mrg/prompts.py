"""Prompt construction (Section 8.4.4).

Every prompt = system message + task section + operator section + population
section. The population section shows code plus one feedback line per scene:
fitness, OA_val, OA_grp, mF1_val, number of regions, mean region size, purity
on training pixels and run time. Failed designs show their error message.
"""
from __future__ import annotations

import inspect

from . import blocks

SYSTEM = (
    "You are an expert in hyperspectral image analysis and in the design of image "
    "segmentation algorithms. You write region-based segmentation methods as Python code. "
    "Reply with exactly one complete Python file inside a single ```python code block and "
    "nothing else outside it."
)

CONTRACT = """\
DESIGN CONTRACT. The file must define:
  DESCRIPTION : str   one sentence stating the core idea of the design
  PARAMS      : dict  every number the design uses (no magic numbers elsewhere)
  CLASSIFIER  : dict  {"kernel": "linear" | "rbf", "C": float, "gamma": float | "scale"}
  LABELING    : "region" | "pixel_vote"
  segment(ctx, params) -> np.ndarray of int, shape (ctx.H, ctx.W): a region map
  region_features(ctx, regions, rows, params) -> np.ndarray (len(rows), d)
      one feature vector per requested region id in `rows` (needed for LABELING="region")
  pixel_features(ctx, params) -> np.ndarray (ctx.H, ctx.W, d)
      needed instead for LABELING="pixel_vote": the SVM classifies pixels and every
      region takes the majority vote of its pixels.

ctx holds unlabelled data only:
  ctx.v      (H, W)      V channel of the first three principal components, in [0, 1]
  ctx.pcs    (H, W, 10)  first ten standardised principal components
  ctx.bands  (H, W, B)   all standardised bands
  ctx.H, ctx.W, ctx.B, ctx.rng (seeded numpy Generator; use it for all randomness)
Labels and the data split are never visible. The harness trains the SVM, so a design
can only improve by producing better regions or better region descriptions.
The same code and PARAMS run on all five scenes; do not branch on the scene.
Allowed imports: numpy, scipy (not scipy.io), skimage, sklearn, numba, math,
itertools, functools, collections, heapq, and `from llm_mrg import blocks as B`.
No file, network or process access, no eval/exec.
"""


def blocks_api() -> str:
    names = ["mrg", "orientation_map", "glcm88", "grow", "histogram_seed_order", "grid_cells",
             "region_means", "region_stds", "region_sizes", "region_adjacency", "merge_small",
             "connected_components", "relabel"]
    lines = ["BUILDING BLOCKS (from llm_mrg import blocks as B). You may also write your own code."]
    for n in names:
        f = getattr(blocks, n)
        doc = (inspect.getdoc(f) or "").split("\n\n")[0].replace("\n", " ")
        lines.append(f"  B.{n}{inspect.signature(f)}\n      {doc}")
    return "\n".join(lines)


def task_section(scene_info: list[dict], seed_src: str, timeout_s: float) -> str:
    sc = "\n".join(f"  - {s['name']}: {s['H']} x {s['W']} pixels, {s['B']} bands, "
                   f"{s['K']} classes, {s['n_labelled']} labelled pixels" for s in scene_info)
    return f"""\
TASK. Design a region-based segmentation method for hyperspectral images. Regions are
labelled by an SVM, and the method is scored on five benchmark scenes:
{sc}

STARTING POINT. Generation 0 is a modified region growing (MRG) method transferred from
MRI to hyperspectral data: seeds in histogram-frequency order, a distance threshold to
the seed, an orientation constraint, growth confined to a 3x3 grid, 88 GLCM features plus
the mean spectrum per region, linear SVM:
```python
{seed_src.strip()}
```

{CONTRACT}
{blocks_api()}

RUN-TIME LIMIT. {timeout_s:.0f} s per scene for segment + features together. Exceeding it,
raising an exception, or returning a wrong shape gives fitness 0.

FITNESS (inside the training split only):
  OA_val, mF1_val : SVM trained on 75% of training pixels, validated on the other 25%
  OA_grp          : 25% of the regions that contain training pixels are held out whole,
                    an SVM trained on the remaining regions predicts them (no shared regions)
  F_scene = (OA_val + OA_grp + 100 * mF1_val) / 3,   F = mean over the five scenes
OA_grp punishes designs that only win because a validation pixel shares a region with a
training pixel; mF1 gives rare classes the same weight as common ones.
"""


OPERATORS = {
    "E1": ("EXPLORE. Below is the whole current population. Write a new design whose core "
           "idea differs from every one of them (for example a different channel to grow on, "
           "seed order, similarity criterion, region post-processing or region description)."),
    "E2": ("COMBINE. Below are two parent designs with their per-scene feedback. Write one design "
           "that merges their strongest parts, guided by where each scores well or badly."),
    "M1": ("REFINE. Below is one parent design with its per-scene feedback. Find its weakest point "
           "in the feedback and change the structure of the algorithm to fix it."),
    "M2": ("TUNE. Below is one parent design with its per-scene feedback. You may change ONLY the "
           "values in PARAMS and CLASSIFIER. Every other line must stay exactly the same."),
}


def feedback_lines(entry: dict) -> str:
    out = []
    for name, r in entry.get("per_scene", {}).items():
        if r.get("error"):
            out.append(f"  {name}: FAILED fitness=0  error: {r['error'][-600:]}")
        else:
            out.append(f"  {name}: fitness={r['fitness']:.2f} OA_val={r['oa_val']:.2f} "
                       f"OA_grp={r['oa_grp']:.2f} mF1_val={r['mf1_val']:.4f} "
                       f"regions={r['n_regions']} mean_size={r['mean_region_size']:.1f} "
                       f"purity={r['purity']:.4f} time={r['seconds']:.1f}s")
    return "\n".join(out)


def design_block(entry: dict, header: str) -> str:
    return (f"### {header} {entry['id']}  (overall fitness {entry['fitness']:.2f})\n"
            f"DESCRIPTION: {entry.get('description', '')}\n"
            f"Feedback per scene:\n{feedback_lines(entry)}\n"
            f"```python\n{entry['src'].strip()}\n```\n")


def build_prompt(op: str, task: str, parents: list[dict], population: list[dict]) -> str:
    if op == "E1":
        pop = "\n".join(design_block(e, "DESIGN") for e in population)
    else:
        pop = "\n".join(design_block(e, "PARENT") for e in parents)
    return f"{task}\nOPERATOR {op}. {OPERATORS[op]}\n\nPOPULATION\n{pop}\nReturn the complete new Python file."
