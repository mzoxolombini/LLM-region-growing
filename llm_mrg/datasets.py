"""Loaders for the five benchmark scenes.

Each loader returns (cube[H, W, B] float32, gt[H, W] int32) where gt == 0 is
unlabelled and classes are 1..K.

Four scenes come from the EHU GIC repository as .mat files. Houston 2013 is
distributed by the IEEE GRSS (Data Fusion Contest 2013) and needs registration,
so it has to be placed in the data directory by hand. Several .mat repackagings
of Houston 2013 circulate; the loader tries common key names and falls back to
the first 3-D / 2-D arrays it finds.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

EHU = "https://www.ehu.eus/ccwintco/uploads"


@dataclass
class SceneSpec:
    name: str
    cube_file: str
    gt_file: str
    cube_keys: tuple
    gt_keys: tuple
    cube_url: str | None = None
    gt_url: str | None = None


SCENES = {
    "indian_pines": SceneSpec(
        "indian_pines", "Indian_pines_corrected.mat", "Indian_pines_gt.mat",
        ("indian_pines_corrected",), ("indian_pines_gt",),
        f"{EHU}/6/67/Indian_pines_corrected.mat", f"{EHU}/c/c4/Indian_pines_gt.mat"),
    "salinas": SceneSpec(
        "salinas", "Salinas_corrected.mat", "Salinas_gt.mat",
        ("salinas_corrected",), ("salinas_gt",),
        f"{EHU}/a/a3/Salinas_corrected.mat", f"{EHU}/f/fa/Salinas_gt.mat"),
    "pavia_university": SceneSpec(
        "pavia_university", "PaviaU.mat", "PaviaU_gt.mat",
        ("paviaU",), ("paviaU_gt",),
        f"{EHU}/e/ee/PaviaU.mat", f"{EHU}/5/50/PaviaU_gt.mat"),
    "botswana": SceneSpec(
        "botswana", "Botswana.mat", "Botswana_gt.mat",
        ("Botswana",), ("Botswana_gt",),
        f"{EHU}/7/72/Botswana.mat", f"{EHU}/5/58/Botswana_gt.mat"),
    # Houston 2013 (CASI, 349 x 1905, 144 bands, 15 classes). Manual download.
    "houston": SceneSpec(
        "houston", "Houston.mat", "Houston_gt.mat",
        ("Houston", "houston", "img", "data", "HSI"),
        ("Houston_gt", "houston_gt", "gt", "GT", "label", "labels"),
        None, None),
}


def _pick(mat: dict, keys: tuple, ndim: int) -> np.ndarray:
    for k in keys:
        if k in mat:
            return np.asarray(mat[k])
    for k, v in mat.items():
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == ndim:
            return v
    raise KeyError(f"No array with keys {keys} or ndim={ndim}; found {list(mat)}")


def _load_mat(path: str) -> dict:
    try:
        return sio.loadmat(path)
    except NotImplementedError:  # MATLAB v7.3 (HDF5)
        import h5py
        with h5py.File(path, "r") as f:
            return {k: np.array(f[k]).T for k in f.keys()}


def download(name: str, data_dir: str) -> None:
    spec = SCENES[name]
    os.makedirs(data_dir, exist_ok=True)
    for fn, url in ((spec.cube_file, spec.cube_url), (spec.gt_file, spec.gt_url)):
        dst = os.path.join(data_dir, fn)
        if os.path.exists(dst):
            continue
        if url is None:
            raise FileNotFoundError(
                f"{dst} missing. {name} must be downloaded manually "
                "(IEEE GRSS DFC 2013 for Houston). See README.")
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dst)


def load_scene(name: str, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    spec = SCENES[name]
    cube = _pick(_load_mat(os.path.join(data_dir, spec.cube_file)), spec.cube_keys, 3)
    gt = _pick(_load_mat(os.path.join(data_dir, spec.gt_file)), spec.gt_keys, 2)
    cube = cube.astype(np.float32)
    gt = gt.astype(np.int32)
    if cube.shape[:2] != gt.shape:
        # some HDF5 repackagings store bands first
        if cube.shape[1:] == gt.shape:
            cube = np.moveaxis(cube, 0, -1)
        else:
            raise ValueError(f"{name}: cube {cube.shape} vs gt {gt.shape}")
    return cube, gt


def make_synthetic(h=60, w=70, bands=30, k=4, seed=0):
    """Small piecewise-constant scene for smoke tests only."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((h, w), np.int32)
    yy, xx = np.mgrid[:h, :w]
    centers = rng.uniform([0, 0], [h, w], size=(k * 3, 2))
    d = np.stack([(yy - c[0]) ** 2 + (xx - c[1]) ** 2 for c in centers], -1)
    lab = np.argmin(d, -1) % k + 1
    spectra = rng.normal(0, 1, (k + 1, bands)).cumsum(1)
    cube = spectra[lab] + rng.normal(0, 0.8, (h, w, bands))
    gt[:] = lab
    gt[rng.random((h, w)) < 0.3] = 0  # some unlabelled pixels
    return cube.astype(np.float32), gt
