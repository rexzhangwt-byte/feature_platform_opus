"""Feature computation engine.

A feature-compute config (Module 2) is a graph of nodes and edges. Nodes
are either:
  - "input"   : references an existing dataset/feature-set by id
  - "op"      : a math operation (op_type field)
  - "output"  : marks a node whose result is exported as a new feature set

Each node receives one or more upstream tensors (dict[str, list[float]])
and produces a tensor of the same shape.

Supported op_type values:
    standardize        : per-row z-score (std=0 -> zeros)
    concat             : concat vectors along the feature dim across inputs
    add, sub, mul, div : pairwise element-wise on aligned keys
    mean               : average across all inputs
    aggregate_mean     : reduce vector to a scalar (mean across dims)
    aggregate_sum      : reduce vector to a scalar (sum across dims)
    aggregate_max      : reduce vector to a scalar (max)
    aggregate_min      : reduce vector to a scalar (min)
    filter_keys        : intersect with a provided allow-list / target keys
    scale              : multiply by a constant
    log1p              : log(1+x) per element
    abs                : absolute value
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _vec(v):
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist()
    if isinstance(v, list):
        return [float(x) for x in v]
    if isinstance(v, (int, float)):
        return [float(v)]
    raise ValueError(f"Unsupported feature value type: {type(v).__name__}")


def _aligned(*dicts: dict[str, Any]) -> tuple[list[str], list[dict[str, list[float]]]]:
    if not dicts:
        return [], []
    keys = set(dicts[0].keys())
    for d in dicts[1:]:
        keys &= set(d.keys())
    keys = sorted(keys)
    aligned = []
    for d in dicts:
        aligned.append({k: _vec(d[k]) for k in keys})
    return keys, aligned


def op_standardize(inp: dict) -> dict:
    out = {}
    for k, v in inp.items():
        arr = np.asarray(_vec(v), dtype=float)
        std = arr.std()
        if std == 0 or not np.isfinite(std):
            out[k] = np.zeros_like(arr).tolist()
        else:
            out[k] = ((arr - arr.mean()) / std).tolist()
    return out


def op_concat(*inps: dict) -> dict:
    keys, aligned = _aligned(*inps)
    out = {}
    for k in keys:
        merged = []
        for d in aligned:
            merged.extend(d[k])
        out[k] = merged
    return out


def _binary(a: dict, b: dict, fn) -> dict:
    keys, [da, db] = _aligned(a, b)
    out = {}
    for k in keys:
        va = np.asarray(da[k], dtype=float)
        vb = np.asarray(db[k], dtype=float)
        L = min(len(va), len(vb))
        out[k] = fn(va[:L], vb[:L]).tolist()
    return out


def op_add(a, b):
    return _binary(a, b, lambda x, y: x + y)


def op_sub(a, b):
    return _binary(a, b, lambda x, y: x - y)


def op_mul(a, b):
    return _binary(a, b, lambda x, y: x * y)


def op_div(a, b):
    return _binary(a, b, lambda x, y: np.divide(x, y, out=np.zeros_like(x),
                                                 where=(y != 0)))


def op_mean(*inps: dict) -> dict:
    keys, aligned = _aligned(*inps)
    out = {}
    for k in keys:
        stacked = np.stack([np.asarray(d[k], dtype=float) for d in aligned])
        out[k] = stacked.mean(axis=0).tolist()
    return out


def op_aggregate(inp: dict, agg: str) -> dict:
    fn = {
        "mean": np.mean, "sum": np.sum, "max": np.max, "min": np.min,
    }[agg]
    out = {}
    for k, v in inp.items():
        arr = np.asarray(_vec(v), dtype=float)
        out[k] = [float(fn(arr))]
    return out


def op_filter_keys(inp: dict, allow_keys: list[str] | None) -> dict:
    if not allow_keys:
        return dict(inp)
    allow = set(str(x) for x in allow_keys)
    return {k: v for k, v in inp.items() if k in allow}


def op_scale(inp: dict, factor: float) -> dict:
    return {k: (np.asarray(_vec(v), dtype=float) * float(factor)).tolist()
            for k, v in inp.items()}


def op_log1p(inp: dict) -> dict:
    return {k: np.log1p(np.maximum(0, np.asarray(_vec(v), dtype=float))).tolist()
            for k, v in inp.items()}


def op_abs(inp: dict) -> dict:
    return {k: np.abs(np.asarray(_vec(v), dtype=float)).tolist()
            for k, v in inp.items()}


def execute_graph(graph: dict, resolve_input_fn) -> dict[str, dict]:
    """Execute a feature-compute graph.

    graph = {
      "nodes": [{"id": str, "type": "input"|"op"|"output",
                 "op_type": str, "params": {...},
                 "ref_id": str, "ref_kind": "datasets"|"features"}, ...],
      "edges": [{"from": id, "to": id, "port": int (default 0)}, ...]
    }

    resolve_input_fn(ref_kind, ref_id) -> dict (loaded feature set / dataset)

    Returns: {output_node_id: dict_feature_set, ...}
    Also returns "_all_": {node_id: dict}  (every node's result, for caching).
    """
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])

    # Build incoming map preserving port order
    incoming: dict[str, list[tuple[int, str]]] = {nid: [] for nid in nodes}
    for e in edges:
        incoming[e["to"]].append((int(e.get("port", 0)), e["from"]))
    for nid in incoming:
        incoming[nid].sort()

    # Topological order
    order: list[str] = []
    visited: set[str] = set()
    temp: set[str] = set()

    def visit(n):
        if n in visited:
            return
        if n in temp:
            raise ValueError(f"Cycle detected at node {n}")
        temp.add(n)
        for _, src in incoming.get(n, []):
            visit(src)
        temp.discard(n)
        visited.add(n)
        order.append(n)

    for nid in nodes:
        visit(nid)

    cache: dict[str, dict] = {}
    for nid in order:
        node = nodes[nid]
        ntype = node["type"]
        if ntype == "input":
            data = resolve_input_fn(node.get("ref_kind", "features"),
                                    node.get("ref_id"))
            cache[nid] = data
        elif ntype == "output":
            srcs = [cache[s] for _, s in incoming[nid]]
            if not srcs:
                raise ValueError(f"Output node {nid} has no input")
            cache[nid] = srcs[0]
        elif ntype == "op":
            op = node.get("op_type")
            srcs = [cache[s] for _, s in incoming[nid]]
            params = node.get("params", {}) or {}
            if op == "standardize":
                if not srcs:
                    raise ValueError("standardize needs 1 input")
                cache[nid] = op_standardize(srcs[0])
            elif op == "concat":
                cache[nid] = op_concat(*srcs)
            elif op in ("add", "sub", "mul", "div"):
                if len(srcs) < 2:
                    raise ValueError(f"{op} needs 2 inputs")
                fn = {"add": op_add, "sub": op_sub,
                      "mul": op_mul, "div": op_div}[op]
                cache[nid] = fn(srcs[0], srcs[1])
            elif op == "mean":
                cache[nid] = op_mean(*srcs)
            elif op in ("aggregate_mean", "aggregate_sum",
                        "aggregate_max", "aggregate_min"):
                kind = op.split("_", 1)[1]
                cache[nid] = op_aggregate(srcs[0], kind)
            elif op == "filter_keys":
                allow = params.get("allow_keys")
                if allow is None and params.get("allow_from_ref"):
                    ref = resolve_input_fn(
                        params.get("allow_ref_kind", "labels"),
                        params["allow_from_ref"]
                    )
                    allow = list(ref.keys())
                cache[nid] = op_filter_keys(srcs[0], allow)
            elif op == "scale":
                cache[nid] = op_scale(srcs[0], float(params.get("factor", 1.0)))
            elif op == "log1p":
                cache[nid] = op_log1p(srcs[0])
            elif op == "abs":
                cache[nid] = op_abs(srcs[0])
            else:
                raise ValueError(f"Unknown op_type: {op}")
        else:
            raise ValueError(f"Unknown node type: {ntype}")

    outputs = {nid: cache[nid] for nid, n in nodes.items()
               if n["type"] == "output"}
    return {"_outputs": outputs, "_all_": cache}
