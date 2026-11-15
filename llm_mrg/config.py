"""Config loading and scene preparation shared by the scripts."""
from __future__ import annotations

import copy

import yaml

from .harness import prepare_scene


def _set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(open(path))
    cfg = copy.deepcopy(cfg)
    for o in overrides or []:
        k, v = o.split("=", 1)
        _set(cfg, k, v)
    return cfg


def prepare_scenes(cfg: dict):
    return [prepare_scene(n, cfg["data_dir"], cfg["cache_dir"], seed=cfg["seed"]) for n in cfg["scenes"]]
