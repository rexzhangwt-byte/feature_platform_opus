"""Data I/O helpers: load/save dict-shaped feature/label sets in json or pkl.

Throughout the platform, a "feature set" or "label set" is canonically a
dict[str, list[float] | str | int | float]. Feature vectors should be
list[float]; label values can be any scalar.
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_dict_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in (".pkl", ".pickle"):
        with open(p, "rb") as f:
            data = pickle.load(f)
    elif suffix in (".npy",):
        arr = np.load(p, allow_pickle=True).item()
        data = arr
    else:
        # try json then pickle
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            with open(p, "rb") as f:
                data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Loaded object is not a dict (got {type(data).__name__}). "
            f"Expected key->value mapping."
        )
    # normalize keys to str
    out = {}
    for k, v in data.items():
        out[str(k)] = v
    return out


def save_dict_file(path: str | Path, data: dict[str, Any], fmt: str = "json") -> None:
    p = Path(path)
    if fmt == "json":
        # try to make json-safe
        safe = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                safe[k] = v.tolist()
            elif isinstance(v, (list, tuple)):
                safe[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x
                           for x in v]
            else:
                safe[k] = v
        with open(p, "w", encoding="utf-8") as f:
            json.dump(safe, f, ensure_ascii=False)
    elif fmt in ("pkl", "pickle"):
        with open(p, "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unknown format {fmt}")


def summarize_dict(data: dict[str, Any], sample_n: int = 5) -> dict:
    """Quick structural summary used by the dataset visualization view."""
    keys = list(data.keys())
    n = len(keys)
    sample_keys = keys[:sample_n]
    sample_values = []
    value_kinds: dict[str, int] = {}
    vec_lens: list[int] = []
    scalar_examples: list[Any] = []

    for k in keys:
        v = data[k]
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, list):
            value_kinds["vector"] = value_kinds.get("vector", 0) + 1
            vec_lens.append(len(v))
        elif isinstance(v, dict):
            value_kinds["dict"] = value_kinds.get("dict", 0) + 1
        elif isinstance(v, (int, float, np.floating, np.integer)):
            value_kinds["number"] = value_kinds.get("number", 0) + 1
            if len(scalar_examples) < 10:
                scalar_examples.append(float(v))
        elif isinstance(v, str):
            value_kinds["string"] = value_kinds.get("string", 0) + 1
            if len(scalar_examples) < 10:
                scalar_examples.append(v)
        else:
            value_kinds[type(v).__name__] = value_kinds.get(type(v).__name__, 0) + 1

    for k in sample_keys:
        v = data[k]
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, list) and len(v) > 12:
            sample_values.append(v[:6] + ["..."] + v[-3:])
        else:
            sample_values.append(v)

    summary = {
        "num_keys": n,
        "value_kinds": value_kinds,
        "sample_keys": sample_keys,
        "sample_values": sample_values,
        "scalar_examples": scalar_examples,
    }
    if vec_lens:
        arr = np.array(vec_lens)
        summary["vector_length"] = {
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "uniform": bool(arr.min() == arr.max()),
        }
        # Stats on the actual numeric content (sample up to 200 vectors)
        sample_for_stats = keys[: min(200, n)]
        flat = []
        for k in sample_for_stats:
            v = data[k]
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, (int, float, np.floating, np.integer)):
                        flat.append(float(x))
        if flat:
            arr2 = np.array(flat)
            summary["value_stats"] = {
                "min": float(arr2.min()),
                "max": float(arr2.max()),
                "mean": float(arr2.mean()),
                "std": float(arr2.std()),
            }
    return summary


def to_matrix(data: dict[str, list[float]],
              keys: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Convert a dict of equal-length vectors to a (N,D) matrix and the row keys.

    If keys is None, uses sorted dict keys (for deterministic ordering).
    Skips entries whose value is not a numeric vector.
    """
    if keys is None:
        keys = sorted(data.keys())
    rows = []
    valid_keys = []
    for k in keys:
        v = data.get(k)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if not isinstance(v, list):
            continue
        try:
            row = [float(x) for x in v]
        except Exception:
            continue
        rows.append(row)
        valid_keys.append(k)
    if not rows:
        return np.zeros((0, 0)), []
    # truncate / pad to common length
    L = min(len(r) for r in rows)
    rows = [r[:L] for r in rows]
    return np.asarray(rows, dtype=np.float64), valid_keys
