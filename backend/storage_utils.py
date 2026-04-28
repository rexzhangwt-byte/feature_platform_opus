"""Storage utilities for the feature mining platform.

All persistent state lives under backend/storage/<kind>/<id>.json (configs)
or backend/storage/<kind>/<id>.<ext> (binary artifacts: features, labels,
models, archives).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

STORAGE_ROOT = Path(__file__).parent / "storage"
KINDS = [
    "datasets",        # raw uploaded / ssh-pulled datasets
    "features",        # computed feature sets (each is a dict id->vector)
    "labels",          # label sets
    "models",          # model algorithm configs (Module 5)
    "configs",         # other configs: feature-compute (M2), feature-select (M4),
                       # feature-vis (M5 outputs cached), hpo (M6)
    "pipelines",       # training task pipelines (Module 7)
    "archives",        # trained model archives & evaluation reports (Module 8)
]


def ensure_dirs() -> None:
    for k in KINDS:
        (STORAGE_ROOT / k).mkdir(parents=True, exist_ok=True)


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


def now_ts() -> float:
    return time.time()


def kind_dir(kind: str) -> Path:
    if kind not in KINDS:
        raise ValueError(f"Unknown storage kind: {kind}")
    return STORAGE_ROOT / kind


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_kind(kind: str) -> list[dict]:
    """List metadata of all entries of a given kind. Each entry has its
    JSON sidecar at <id>.meta.json."""
    out: list[dict] = []
    d = kind_dir(kind)
    for p in sorted(d.glob("*.meta.json")):
        try:
            out.append(read_json(p))
        except Exception:
            continue
    out.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return out


def get_meta(kind: str, item_id: str) -> dict | None:
    p = kind_dir(kind) / f"{item_id}.meta.json"
    if not p.exists():
        return None
    return read_json(p)


def save_meta(kind: str, item_id: str, meta: dict) -> None:
    p = kind_dir(kind) / f"{item_id}.meta.json"
    write_json(p, meta)


def delete_item(kind: str, item_id: str) -> bool:
    d = kind_dir(kind)
    found = False
    for p in d.glob(f"{item_id}.*"):
        try:
            p.unlink()
            found = True
        except Exception:
            pass
    return found


ensure_dirs()
