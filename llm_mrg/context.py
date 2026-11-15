"""The unlabelled context object handed to every design.

ctx carries only unlabelled data (Section 8.4.1):
  ctx.v      : V channel of the first three principal components  (H, W)
  ctx.pcs    : first ten standardised principal components         (H, W, 10)
  ctx.bands  : all standardised bands                              (H, W, B)
  ctx.H, ctx.W, ctx.B
  ctx.rng    : seeded numpy Generator

ASSUMPTION (Section 6.3.3 is not in the chapter supplied): the V channel is
built by min-max scaling PC1..PC3 to [0, 1], treating them as R, G, B and
taking HSV value V = max(R, G, B). Change `v_channel` if the thesis says
otherwise.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def standardise(x: np.ndarray) -> np.ndarray:
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


def v_channel(pc3: np.ndarray) -> np.ndarray:
    lo = pc3.min(0, keepdims=True)
    hi = pc3.max(0, keepdims=True)
    rgb = (pc3 - lo) / np.where(hi > lo, hi - lo, 1.0)
    return rgb.max(1)


def build_context_arrays(cube: np.ndarray, n_pcs: int = 10, seed: int = 0) -> dict:
    h, w, b = cube.shape
    x = standardise(cube.reshape(-1, b).astype(np.float64))
    n_pcs = min(n_pcs, b)
    pcs = PCA(n_components=n_pcs, random_state=seed, svd_solver="full").fit_transform(x)
    v = v_channel(pcs[:, :3])
    pcs = standardise(pcs)
    return {
        "v": v.reshape(h, w).astype(np.float32),
        "pcs": pcs.reshape(h, w, n_pcs).astype(np.float32),
        "bands": x.reshape(h, w, b).astype(np.float32),
    }


class Context:
    """Read-only view passed to designs. No labels, no split information."""

    def __init__(self, arrays: dict, seed: int, name: str = ""):
        self.v = arrays["v"]
        self.pcs = arrays["pcs"]
        self.bands = arrays["bands"]
        for a in (self.v, self.pcs, self.bands):
            a.setflags(write=False)
        self.H, self.W = self.v.shape
        self.B = self.bands.shape[2]
        self.name = name  # empty inside the sandbox
        self.rng = np.random.default_rng(seed)
        self.trace: list[str] = []  # building blocks record their use here


def save_context(path: str, arrays: dict) -> None:
    np.savez(path, **arrays)


def load_context(path: str, seed: int, name: str = "") -> Context:
    with np.load(path) as z:
        arrays = {k: z[k] for k in ("v", "pcs", "bands")}
    return Context(arrays, seed, name)
