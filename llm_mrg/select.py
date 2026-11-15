"""Choosing LLM-MRG and LLM-RB from a finished search (Sections 8.1 and 5.3).

LLM-RB  : best design of the whole (unconstrained) search.
LLM-MRG : best design that still forms regions with the MRG growth rule:
          (i) histogram-frequency seed order, (ii) distance threshold to the seed,
          (iii) the orientation constraint, (iv) growth inside the 3x3 grid.

The check reads the building-block trace recorded on every scene. A design is
MRG-family only if, on every scene, the trace contains a call to B.mrg with
grid=3 and, under the strict rule, no other block that forms or rewrites
regions. Under the lenient rule, merge_small / connected_components after MRG
are allowed. The trace cannot prove that the design actually returned the MRG
output, so audit the chosen design's code by hand before reporting it.

The chapter defers the exact rule to Section 5.3 (not supplied). Here the rule
is "highest search fitness", with ties broken by the earlier design.
"""
from __future__ import annotations

import json
import os

REGION_OPS_STRICT = ("grow(", "merge_small", "connected_components")
REGION_OPS_LENIENT = ("grow(",)


def segment_returns_mrg(src: str) -> bool:
    """True if segment() returns the output of B.mrg(...) directly. The building-block trace
    cannot see region operations from other libraries (e.g. skimage watershed), so the trace
    test alone admits designs that intersect MRG regions with other partitions."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "segment":
            rets = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            return bool(rets) and all(
                isinstance(r.value, ast.Call) and isinstance(r.value.func, ast.Attribute)
                and r.value.func.attr == "mrg" for r in rets)
    return False


def is_mrg_family(entry: dict, strict: bool = True) -> bool:
    banned = REGION_OPS_STRICT if strict else REGION_OPS_LENIENT
    per = entry.get("per_scene", {})
    if not per:
        return False
    if strict and "path" in entry and not segment_returns_mrg(open(entry["path"]).read()):
        return False
    for r in per.values():
        tr = r.get("trace") or []
        if r.get("error") or not tr:
            return False
        if not any(t.startswith("mrg(") and "grid=3," in t and "orientation=on" in t for t in tr):
            return False
        if any(t.startswith(banned) for t in tr):
            return False
    return True


def load_archive(run_dir: str) -> list[dict]:
    ddir = os.path.join(run_dir, "designs")
    out = []
    for f in sorted(os.listdir(ddir)):
        if f.endswith(".json"):
            e = json.load(open(os.path.join(ddir, f)))
            e["path"] = os.path.join(ddir, f[:-5] + ".py")
            out.append(e)
    return out


def choose(run_dir: str, strict: bool = True) -> dict:
    arch = load_archive(run_dir)
    key = lambda e: (-e["fitness"], e["gen"], e["id"])
    arch.sort(key=key)
    rb = arch[0] if arch else None
    # D0 is itself MRG-family; if it comes out on top the search found no MRG improvement
    mrg = next((e for e in arch if is_mrg_family(e, strict)), None)
    d0 = next((e for e in arch if e["op"] == "D0"), None)
    return {"LLM-RB": rb, "LLM-MRG": mrg, "Baseline MRG": d0,
            "n_designs": len(arch), "n_mrg_family": sum(is_mrg_family(e, strict) for e in arch)}
