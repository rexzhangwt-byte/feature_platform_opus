"""Feature selection: low-quality filtering + importance/weight strategies.

A feature-select config has the schema:
{
  "filters": [
     {"type": "value_range", "column": int|"all",
      "min": float|null, "max": float|null},
     {"type": "variance", "min_var": float},   # remove cols with var < min_var
     {"type": "missing_drop"},                 # drop rows with NaN/inf
     {"type": "topk_variance", "k": int}       # keep top-k variance columns
  ],
  "importance": {
     "strategy": "manual" | "variance" | "uniform",
     "weights": [float, ...]   # for "manual"
     "priorities": [int, ...]  # optional priority vector (higher first)
  }
}
"""
from __future__ import annotations

import numpy as np


def apply_filters(matrix: np.ndarray,
                  keys: list[str],
                  filters: list[dict]) -> tuple[np.ndarray, list[str], list[int], dict]:
    """Returns the filtered matrix, surviving keys, surviving column indices and report."""
    if matrix.size == 0:
        return matrix, list(keys), list(range(matrix.shape[1] if matrix.ndim == 2 else 0)), {"steps": []}
    M = matrix.copy()
    K = list(keys)
    cols = list(range(M.shape[1]))
    report = {"steps": [], "input_shape": list(M.shape)}

    for f in filters or []:
        ft = f.get("type")
        before_shape = list(M.shape)
        if ft == "value_range":
            col = f.get("column", "all")
            lo = f.get("min", None)
            hi = f.get("max", None)
            mask = np.ones(M.shape[0], dtype=bool)
            if col == "all" or col is None:
                if lo is not None:
                    mask &= (M >= lo).all(axis=1)
                if hi is not None:
                    mask &= (M <= hi).all(axis=1)
            else:
                ci = int(col)
                if 0 <= ci < M.shape[1]:
                    if lo is not None:
                        mask &= M[:, ci] >= lo
                    if hi is not None:
                        mask &= M[:, ci] <= hi
            M = M[mask]
            K = [K[i] for i, m in enumerate(mask) if m]
            report["steps"].append({"filter": "value_range", "before": before_shape,
                                    "after": list(M.shape)})
        elif ft == "variance":
            min_var = float(f.get("min_var", 0.0))
            if M.shape[0] > 1:
                var = M.var(axis=0)
                keep = var >= min_var
                M = M[:, keep]
                cols = [c for c, k in zip(cols, keep) if k]
                report["steps"].append({"filter": "variance", "min_var": min_var,
                                        "kept_columns": int(keep.sum()),
                                        "dropped_columns": int((~keep).sum())})
        elif ft == "missing_drop":
            mask = np.isfinite(M).all(axis=1)
            M = M[mask]
            K = [K[i] for i, m in enumerate(mask) if m]
            report["steps"].append({"filter": "missing_drop", "kept_rows": int(mask.sum()),
                                    "dropped_rows": int((~mask).sum())})
        elif ft == "topk_variance":
            k = int(f.get("k", M.shape[1]))
            if M.shape[0] > 1 and k < M.shape[1]:
                var = M.var(axis=0)
                top = np.argsort(-var)[:k]
                top_sorted = sorted(top.tolist())
                M = M[:, top_sorted]
                cols = [cols[i] for i in top_sorted]
                report["steps"].append({"filter": "topk_variance", "k": k})
        else:
            report["steps"].append({"filter": ft, "skipped": "unknown"})

    report["output_shape"] = list(M.shape)
    return M, K, cols, report


def apply_importance(matrix: np.ndarray, importance_cfg: dict) -> tuple[np.ndarray, list[float]]:
    """Returns the (re-weighted) matrix and the weight vector actually applied."""
    if matrix.size == 0 or matrix.ndim != 2:
        return matrix, []
    n_cols = matrix.shape[1]
    strategy = (importance_cfg or {}).get("strategy", "uniform")
    if strategy == "manual":
        w = list((importance_cfg or {}).get("weights") or [])
        if len(w) < n_cols:
            w = list(w) + [1.0] * (n_cols - len(w))
        elif len(w) > n_cols:
            w = w[:n_cols]
        w = [float(x) for x in w]
    elif strategy == "variance":
        var = matrix.var(axis=0)
        s = var.sum() or 1.0
        w = (var / s * n_cols).tolist()
    else:  # uniform
        w = [1.0] * n_cols
    arr = matrix * np.asarray(w, dtype=float)[None, :]
    return arr, w
