"""Isolation and failure handling (Section 8.4.6).

Every design runs on every scene in its own spawned process with a wall-clock
limit. The child gets the path of an unlabelled context file only. The parent
holds the labels and talks to the child through a pipe:

    child  -> ("regions", region_map, trace, seconds, LABELING, CLASSIFIER)
    parent -> rows        (region ids it wants described)  | pixel ids in pixel_vote mode
    child  -> ("features", matrix, seconds)
    child  -> ("error", message)   at any point

A static AST check runs before anything is executed. It blocks file, network
and process access and dynamic code execution. This is a guard against
accidents and casual leakage, not a security boundary: run the search inside a
container (or at least as a user who cannot read the label files) if the LLM is
not trusted. The label .npy files are never written next to the context files.
"""
from __future__ import annotations

import ast
import multiprocessing as mp
import time
import traceback
import types

import numpy as np

ALLOWED_IMPORTS = {
    "numpy", "scipy", "skimage", "sklearn", "numba", "math", "itertools", "functools",
    "collections", "heapq", "typing", "dataclasses", "llm_mrg", "llm_mrg.blocks",
    "warnings",
}
BLOCKED_MODULE_PREFIX = ("scipy.io", "skimage.io", "sklearn.datasets", "numpy.lib.npyio")
BLOCKED_NAMES = {"open", "eval", "exec", "compile", "__import__", "globals", "locals",
                 "vars", "input", "breakpoint", "exit", "quit", "memoryview"}
BLOCKED_ATTRS = {"load", "loadtxt", "fromfile", "genfromtxt", "save", "savez",
                 "savez_compressed", "savetxt", "tofile", "memmap", "system", "popen",
                 "imread", "imsave", "loadmat", "savemat", "fetch_openml"}


def static_check(src: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    errs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                root = m.split(".")[0]
                if m.startswith("llm_mrg") and m not in ("llm_mrg", "llm_mrg.blocks"):
                    errs.append(f"import of {m} is not allowed (only llm_mrg.blocks)")
                elif root not in ALLOWED_IMPORTS or m.startswith(BLOCKED_MODULE_PREFIX):
                    errs.append(f"import of {m} is not allowed")
            if isinstance(node, ast.ImportFrom) and node.module == "llm_mrg":
                for a in node.names:
                    if a.name != "blocks":
                        errs.append(f"from llm_mrg import {a.name} is not allowed")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            errs.append(f"use of {node.id} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRS:
                errs.append(f"attribute .{node.attr} is not allowed")
            if node.attr.startswith("__") and node.attr not in ("__init__", "__name__"):
                errs.append(f"dunder attribute .{node.attr} is not allowed")
    return sorted(set(errs))


def load_design(src: str, name: str = "design") -> types.ModuleType:
    mod = types.ModuleType(name)
    exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    return mod


def check_contract(mod) -> list[str]:
    errs = []
    if not isinstance(getattr(mod, "DESCRIPTION", None), str):
        errs.append("DESCRIPTION must be a string")
    if not isinstance(getattr(mod, "PARAMS", None), dict):
        errs.append("PARAMS must be a dict")
    clf = getattr(mod, "CLASSIFIER", None)
    if not isinstance(clf, dict) or clf.get("kernel") not in ("linear", "rbf"):
        errs.append("CLASSIFIER must be a dict with kernel 'linear' or 'rbf'")
    lab = getattr(mod, "LABELING", None)
    if lab not in ("region", "pixel_vote"):
        errs.append("LABELING must be 'region' or 'pixel_vote'")
    if not callable(getattr(mod, "segment", None)):
        errs.append("segment(ctx, params) is missing")
    if lab == "region" and not callable(getattr(mod, "region_features", None)):
        errs.append("region_features(ctx, regions, rows, params) is missing")
    if lab == "pixel_vote" and not callable(getattr(mod, "pixel_features", None)):
        errs.append("pixel_features(ctx, params) is missing for pixel_vote")
    return errs


def _short_tb(limit=1500):
    tb = traceback.format_exc()
    return tb[-limit:]


def _worker(src, ctx_path, seed, scene, conn, n_threads):
    try:
        import os
        os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
        try:
            import numba
            numba.set_num_threads(max(1, min(n_threads, numba.config.NUMBA_NUM_THREADS)))
        except Exception:
            pass
        from llm_mrg.context import load_context
        from llm_mrg.blocks import relabel
        ctx = load_context(ctx_path, seed, "")  # scene name withheld so designs cannot branch on it
        mod = load_design(src)
        errs = check_contract(mod)
        if errs:
            conn.send(("error", "; ".join(errs)))
            return
        t0 = time.time()
        reg = np.asarray(mod.segment(ctx, dict(mod.PARAMS)))
        if reg.shape != (ctx.H, ctx.W):
            conn.send(("error", f"segment returned shape {reg.shape}, expected {(ctx.H, ctx.W)}"))
            return
        if not np.issubdtype(reg.dtype, np.integer):
            if not np.all(np.isfinite(reg)) or not np.all(reg == np.round(reg)):
                conn.send(("error", "segment must return an integer region map"))
                return
        reg = relabel(reg.astype(np.int64))
        t_seg = time.time() - t0
        clf = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in dict(mod.CLASSIFIER).items()}
        conn.send(("regions", reg, list(ctx.trace), t_seg, mod.LABELING, clf))
        req = conn.recv()
        if req is None:
            return
        t0 = time.time()
        if mod.LABELING == "region":
            F = np.asarray(mod.region_features(ctx, reg, np.asarray(req), dict(mod.PARAMS)), np.float64)
            if F.ndim != 2 or F.shape[0] != len(req):
                conn.send(("error", f"region_features returned {F.shape}, expected ({len(req)}, d)"))
                return
        else:
            P = np.asarray(mod.pixel_features(ctx, dict(mod.PARAMS)), np.float64)
            if P.shape[:2] == (ctx.H, ctx.W):
                P = P.reshape(ctx.H * ctx.W, -1)
            if P.ndim != 2 or P.shape[0] != ctx.H * ctx.W:
                conn.send(("error", f"pixel_features returned {P.shape}, expected (H, W, d) or (H*W, d)"))
                return
            F = P[np.asarray(req)]
        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        conn.send(("features", F, time.time() - t0))
    except BaseException:
        try:
            conn.send(("error", _short_tb()))
        except Exception:
            pass
    finally:
        conn.close()


class DesignRunError(RuntimeError):
    pass


def run_isolated(src: str, ctx_path: str, seed: int, scene: str, request_fn,
                 timeout: float = 600.0, n_threads: int = 1):
    """Run a design on one scene. request_fn(regions, labeling) -> ids to describe.
    Returns (regions, ids, features, trace, labeling, classifier, seconds).
    Raises DesignRunError."""
    errs = static_check(src)
    if errs:
        raise DesignRunError("static check failed: " + "; ".join(errs))
    mpctx = mp.get_context("spawn")
    parent, child = mpctx.Pipe()
    p = mpctx.Process(target=_worker, args=(src, ctx_path, seed, scene, child, n_threads), daemon=True)
    t_start = time.time()
    p.start()
    child.close()

    def _recv():
        left = timeout - (time.time() - t_start)
        if left <= 0 or not parent.poll(left):
            raise DesignRunError(f"time-out after {timeout:.0f} s")
        try:
            return parent.recv()
        except EOFError:
            raise DesignRunError(f"process died (exit code {p.exitcode}); possibly out of memory")

    try:
        msg = _recv()
        if msg[0] == "error":
            raise DesignRunError(msg[1])
        _, regions, trace, _, labeling, clf = msg
        ids = request_fn(regions, labeling)
        parent.send(np.asarray(ids, np.int64))
        msg = _recv()
        if msg[0] == "error":
            raise DesignRunError(msg[1])
        feats = msg[1]
        return regions, ids, feats, trace, labeling, clf, time.time() - t_start
    finally:
        if p.is_alive():
            p.kill()
        p.join(5)
        parent.close()
