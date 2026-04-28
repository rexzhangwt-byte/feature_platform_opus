"""FastAPI application: exposes all 8 modules via REST.

Endpoints (overview):

  Module 1 — Datasets
    POST   /datasets/upload            (multipart)  upload local file
    POST   /datasets/ssh               pull via SSH
    GET    /datasets                   list
    GET    /datasets/{id}/structure    structural visualization payload
    DELETE /datasets/{id}

  Module 2 — Feature Compute
    GET    /feature_ops                list available math ops
    POST   /feature_compute            create config (name + graph)
    GET    /feature_compute            list
    GET    /feature_compute/{id}
    POST   /feature_compute/{id}/run   execute graph -> save outputs as feature sets
    DELETE /feature_compute/{id}

  Direct feature sets
    POST   /features/from_dataset      promote a dataset directly to feature set
    GET    /features                   list
    GET    /features/{id}/structure
    GET    /features/{id}/download     download as json
    DELETE /features/{id}

  Module 3 — Labels
    POST   /labels/upload
    GET    /labels
    GET    /labels/{id}/distribution
    DELETE /labels/{id}

  Module 4 — Feature Selection
    POST   /feature_select             create config
    GET    /feature_select
    GET    /feature_select/{id}
    POST   /feature_select/{id}/preview   apply on a chosen feature set
    DELETE /feature_select/{id}

  Module 5 (visualization helper)
    POST   /viz/single                 single-key time series
    POST   /viz/distribution           overall distribution

  Module 6 — Models
    GET    /algos                      list algorithms + their schemas
    POST   /models                     create a model config (name+algo+params)
    GET    /models
    GET    /models/{id}
    DELETE /models/{id}

  Module 7 — HPO
    POST   /hpo                        create HPO config
    GET    /hpo
    GET    /hpo/{id}
    DELETE /hpo/{id}

  Module 8 — Pipelines
    POST   /pipelines                  create training pipeline
    GET    /pipelines
    GET    /pipelines/{id}
    POST   /pipelines/{id}/run         run pipeline (background)
    GET    /pipelines/{id}/run_status  poll status
    DELETE /pipelines/{id}

  Module 9 — Archives
    GET    /archives                   list
    GET    /archives/{id}
    GET    /archives/{id}/file/{name}  download specific artifact
    DELETE /archives/{id}
"""
from __future__ import annotations

import shutil
import threading
import traceback
from pathlib import Path

import numpy as np
from fastapi import (FastAPI, File, Form, HTTPException, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import data_io, feature_compute, feature_select as fsel
from . import models as mdl
from . import storage_utils as su
from . import ssh_loader, training

app = FastAPI(title="Feature Mining Platform", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ============================================================
# Helpers
# ============================================================

def _common_meta(name: str, kind: str, **extra) -> dict:
    return {"id": su.new_id(), "name": name, "kind": kind,
            "created_at": su.now_ts(), **extra}


def _save_dict_artifact(kind: str, item_id: str, data: dict, fmt: str = "json") -> str:
    ext = "json" if fmt == "json" else "pkl"
    p = su.kind_dir(kind) / f"{item_id}.{ext}"
    data_io.save_dict_file(p, data, fmt=fmt)
    return str(p)


# In-memory pipeline run status
_RUN_STATUS: dict[str, dict] = {}


# ============================================================
# Module 1 — Datasets
# ============================================================

@app.post("/datasets/upload")
async def datasets_upload(file: UploadFile = File(...),
                          name: str = Form(...)):
    item_id = su.new_id("ds_")
    ext = Path(file.filename or "data.json").suffix.lower() or ".json"
    if ext not in (".json", ".pkl", ".pickle", ".npy"):
        raise HTTPException(400, f"Unsupported extension: {ext}")
    out_path = su.kind_dir("datasets") / f"{item_id}{ext}"
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        loaded = data_io.load_dict_file(out_path)
    except Exception as e:
        out_path.unlink(missing_ok=True)
        raise HTTPException(400, f"File could not be parsed as a dict: {e}")
    summary = data_io.summarize_dict(loaded)
    meta = _common_meta(name, "datasets", id=item_id,
                        path=str(out_path),
                        source="upload",
                        original_filename=file.filename,
                        summary=summary)
    su.save_meta("datasets", item_id, meta)
    return meta


class SSHFetchReq(BaseModel):
    name: str
    host: str
    port: int = 22
    username: str
    password: str | None = None
    private_key: str | None = None
    remote_path: str


@app.post("/datasets/ssh")
async def datasets_ssh(req: SSHFetchReq):
    item_id = su.new_id("ds_")
    ext = Path(req.remote_path).suffix.lower() or ".json"
    if ext not in (".json", ".pkl", ".pickle", ".npy"):
        raise HTTPException(400, f"Unsupported remote extension: {ext}")
    out_path = su.kind_dir("datasets") / f"{item_id}{ext}"
    try:
        info = ssh_loader.fetch_via_ssh(
            req.host, req.port, req.username, req.password, req.private_key,
            req.remote_path, out_path,
        )
    except Exception as e:
        raise HTTPException(400, f"SSH fetch failed: {e}")
    try:
        loaded = data_io.load_dict_file(out_path)
    except Exception as e:
        out_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Remote file could not be parsed: {e}")
    summary = data_io.summarize_dict(loaded)
    meta = _common_meta(req.name, "datasets", id=item_id,
                        path=str(out_path),
                        source="ssh",
                        ssh_info={"host": req.host, "port": req.port,
                                  "username": req.username,
                                  "remote_path": req.remote_path,
                                  "size": info["size"]},
                        summary=summary)
    su.save_meta("datasets", item_id, meta)
    return meta


@app.get("/datasets")
def datasets_list():
    return su.list_kind("datasets")


@app.get("/datasets/{ds_id}/structure")
def datasets_structure(ds_id: str):
    meta = su.get_meta("datasets", ds_id)
    if not meta:
        raise HTTPException(404, "dataset not found")
    data = data_io.load_dict_file(meta["path"])
    summary = data_io.summarize_dict(data, sample_n=8)
    return {"meta": meta, "summary": summary}


@app.delete("/datasets/{ds_id}")
def datasets_delete(ds_id: str):
    return {"deleted": su.delete_item("datasets", ds_id)}


# ============================================================
# Promote a dataset to feature set
# ============================================================

class PromoteReq(BaseModel):
    dataset_id: str
    name: str


@app.post("/features/from_dataset")
def features_from_dataset(req: PromoteReq):
    ds_meta = su.get_meta("datasets", req.dataset_id)
    if not ds_meta:
        raise HTTPException(404, "dataset not found")
    data = data_io.load_dict_file(ds_meta["path"])
    fid = su.new_id("fs_")
    path = _save_dict_artifact("features", fid, data, fmt="json")
    summary = data_io.summarize_dict(data)
    meta = _common_meta(req.name, "features", id=fid, path=path,
                        source="dataset", source_dataset_id=req.dataset_id,
                        summary=summary)
    su.save_meta("features", fid, meta)
    return meta


@app.get("/features")
def features_list():
    return su.list_kind("features")


@app.get("/features/{fid}/structure")
def features_structure(fid: str):
    meta = su.get_meta("features", fid)
    if not meta:
        raise HTTPException(404, "feature set not found")
    data = data_io.load_dict_file(meta["path"])
    summary = data_io.summarize_dict(data, sample_n=8)
    return {"meta": meta, "summary": summary}


@app.get("/features/{fid}/download")
def features_download(fid: str):
    meta = su.get_meta("features", fid)
    if not meta:
        raise HTTPException(404, "feature set not found")
    return FileResponse(meta["path"], filename=f"{meta['name']}.json")


@app.delete("/features/{fid}")
def features_delete(fid: str):
    return {"deleted": su.delete_item("features", fid)}


# ============================================================
# Module 2 — Feature compute graph
# ============================================================

@app.get("/feature_ops")
def feature_ops():
    return [
        {"op": "standardize", "n_in": 1,
         "desc": "Per-row z-score; std=0 → zeros (matches mall task spec)"},
        {"op": "concat", "n_in": "*",
         "desc": "Concatenate vectors along feature dim across inputs"},
        {"op": "add", "n_in": 2, "desc": "Element-wise add (a + b)"},
        {"op": "sub", "n_in": 2, "desc": "Element-wise subtract (a - b)"},
        {"op": "mul", "n_in": 2, "desc": "Element-wise multiply (a * b)"},
        {"op": "div", "n_in": 2, "desc": "Element-wise divide (a / b, 0-safe)"},
        {"op": "mean", "n_in": "*", "desc": "Element-wise mean of all inputs"},
        {"op": "aggregate_mean", "n_in": 1, "desc": "Reduce vector → scalar mean"},
        {"op": "aggregate_sum", "n_in": 1, "desc": "Reduce vector → scalar sum"},
        {"op": "aggregate_max", "n_in": 1, "desc": "Reduce vector → scalar max"},
        {"op": "aggregate_min", "n_in": 1, "desc": "Reduce vector → scalar min"},
        {"op": "filter_keys", "n_in": 1,
         "desc": "Keep only keys present in `params.allow_keys` or in a referenced label/feature set"},
        {"op": "scale", "n_in": 1, "desc": "Multiply by params.factor"},
        {"op": "log1p", "n_in": 1, "desc": "log(1+x), clipped at 0"},
        {"op": "abs", "n_in": 1, "desc": "Absolute value"},
    ]


class ComputeCfgReq(BaseModel):
    name: str
    graph: dict


@app.post("/feature_compute")
def feature_compute_create(req: ComputeCfgReq):
    cid = su.new_id("fc_")
    meta = _common_meta(req.name, "configs", id=cid,
                        config_kind="feature_compute",
                        graph=req.graph)
    su.save_meta("configs", cid, meta)
    return meta


@app.get("/feature_compute")
def feature_compute_list():
    return [m for m in su.list_kind("configs")
            if m.get("config_kind") == "feature_compute"]


@app.get("/feature_compute/{cid}")
def feature_compute_get(cid: str):
    meta = su.get_meta("configs", cid)
    if not meta or meta.get("config_kind") != "feature_compute":
        raise HTTPException(404, "feature_compute config not found")
    return meta


def _resolve_input(ref_kind: str, ref_id: str) -> dict:
    meta = su.get_meta(ref_kind, ref_id)
    if not meta:
        raise ValueError(f"{ref_kind}/{ref_id} not found")
    return data_io.load_dict_file(meta["path"])


class RunComputeReq(BaseModel):
    output_names: dict[str, str] | None = None


@app.post("/feature_compute/{cid}/run")
def feature_compute_run(cid: str, req: RunComputeReq):
    meta = su.get_meta("configs", cid)
    if not meta or meta.get("config_kind") != "feature_compute":
        raise HTTPException(404, "feature_compute config not found")
    try:
        result = feature_compute.execute_graph(meta["graph"], _resolve_input)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, f"Graph execution failed: {e}")
    outputs = result["_outputs"]
    saved = []
    for node_id, data in outputs.items():
        name = (req.output_names or {}).get(
            node_id, f"{meta['name']}__{node_id}")
        fid = su.new_id("fs_")
        path = _save_dict_artifact("features", fid, data, fmt="json")
        summary = data_io.summarize_dict(data)
        fmeta = _common_meta(name, "features", id=fid, path=path,
                             source="compute",
                             source_compute_id=cid, source_node=node_id,
                             summary=summary)
        su.save_meta("features", fid, fmeta)
        saved.append(fmeta)
    return {"feature_sets": saved}


@app.delete("/feature_compute/{cid}")
def feature_compute_delete(cid: str):
    return {"deleted": su.delete_item("configs", cid)}


# ============================================================
# Module 3 — Labels
# ============================================================

@app.post("/labels/upload")
async def labels_upload(file: UploadFile = File(...), name: str = Form(...)):
    item_id = su.new_id("lb_")
    ext = Path(file.filename or "labels.json").suffix.lower() or ".json"
    if ext not in (".json", ".pkl", ".pickle"):
        raise HTTPException(400, f"Unsupported extension: {ext}")
    out_path = su.kind_dir("labels") / f"{item_id}{ext}"
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        loaded = data_io.load_dict_file(out_path)
    except Exception as e:
        out_path.unlink(missing_ok=True)
        raise HTTPException(400, f"File could not be parsed: {e}")

    counts: dict[str, int] = {}
    for v in loaded.values():
        counts[str(v)] = counts.get(str(v), 0) + 1
    meta = _common_meta(name, "labels", id=item_id, path=str(out_path),
                        source="upload",
                        original_filename=file.filename,
                        n_samples=len(loaded),
                        distribution=counts)
    su.save_meta("labels", item_id, meta)
    return meta


@app.get("/labels")
def labels_list():
    return su.list_kind("labels")


@app.get("/labels/{lid}/distribution")
def labels_distribution(lid: str):
    meta = su.get_meta("labels", lid)
    if not meta:
        raise HTTPException(404, "label set not found")
    data = data_io.load_dict_file(meta["path"])
    counts: dict[str, int] = {}
    for v in data.values():
        counts[str(v)] = counts.get(str(v), 0) + 1
    return {"meta": meta, "distribution": counts,
            "n_samples": len(data),
            "sample": list(data.items())[:8]}


@app.delete("/labels/{lid}")
def labels_delete(lid: str):
    return {"deleted": su.delete_item("labels", lid)}


# ============================================================
# Module 4 — Feature Selection
# ============================================================

class FSelectReq(BaseModel):
    name: str
    config: dict


@app.post("/feature_select")
def feature_select_create(req: FSelectReq):
    cid = su.new_id("fsel_")
    meta = _common_meta(req.name, "configs", id=cid,
                        config_kind="feature_select",
                        config=req.config)
    su.save_meta("configs", cid, meta)
    return meta


@app.get("/feature_select")
def feature_select_list():
    return [m for m in su.list_kind("configs")
            if m.get("config_kind") == "feature_select"]


@app.get("/feature_select/{cid}")
def feature_select_get(cid: str):
    meta = su.get_meta("configs", cid)
    if not meta or meta.get("config_kind") != "feature_select":
        raise HTTPException(404, "feature_select config not found")
    return meta


class FSelPreviewReq(BaseModel):
    feature_set_id: str


@app.post("/feature_select/{cid}/preview")
def feature_select_preview(cid: str, req: FSelPreviewReq):
    cfg_meta = su.get_meta("configs", cid)
    if not cfg_meta or cfg_meta.get("config_kind") != "feature_select":
        raise HTTPException(404, "feature_select config not found")
    fs_meta = su.get_meta("features", req.feature_set_id)
    if not fs_meta:
        raise HTTPException(404, "feature set not found")
    data = data_io.load_dict_file(fs_meta["path"])
    X, keys = data_io.to_matrix(data)
    cfg = cfg_meta["config"]
    Xf, kf, cols, report = fsel.apply_filters(X, keys, cfg.get("filters", []))
    Xw, weights = fsel.apply_importance(Xf, cfg.get("importance", {}))
    return {
        "before_shape": list(X.shape),
        "after_shape": list(Xw.shape),
        "kept_keys": kf[:20],
        "kept_columns": cols[:20],
        "weights_preview": weights[:20],
        "filter_report": report,
    }


@app.delete("/feature_select/{cid}")
def feature_select_delete(cid: str):
    return {"deleted": su.delete_item("configs", cid)}


# ============================================================
# Visualization helpers (Module 5)
# ============================================================

class VizSingleReq(BaseModel):
    feature_set_id: str
    key: str


@app.post("/viz/single")
def viz_single(req: VizSingleReq):
    fs_meta = su.get_meta("features", req.feature_set_id)
    if not fs_meta:
        raise HTTPException(404, "feature set not found")
    data = data_io.load_dict_file(fs_meta["path"])
    if req.key not in data:
        raise HTTPException(404, f"key {req.key} not in feature set")
    v = data[req.key]
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if not isinstance(v, list):
        v = [float(v)]
    return {"key": req.key, "values": [float(x) for x in v],
            "n": len(v), "name": fs_meta["name"]}


class VizDistReq(BaseModel):
    feature_set_id: str
    column: int = 0
    bins: int = 30


@app.post("/viz/distribution")
def viz_distribution(req: VizDistReq):
    fs_meta = su.get_meta("features", req.feature_set_id)
    if not fs_meta:
        raise HTTPException(404, "feature set not found")
    data = data_io.load_dict_file(fs_meta["path"])
    X, keys = data_io.to_matrix(data)
    if X.size == 0:
        return {"hist": [], "edges": [], "stats": {}, "n_columns": 0}
    col = max(0, min(req.column, X.shape[1] - 1))
    vals = X[:, col]
    hist, edges = np.histogram(vals, bins=int(req.bins))
    col_mean = X.mean(axis=0).tolist()
    col_std = X.std(axis=0).tolist()
    return {
        "column": col,
        "n_columns": X.shape[1],
        "n_samples": X.shape[0],
        "hist": hist.tolist(),
        "edges": edges.tolist(),
        "stats": {"min": float(vals.min()), "max": float(vals.max()),
                  "mean": float(vals.mean()), "std": float(vals.std())},
        "column_mean": col_mean,
        "column_std": col_std,
    }


# ============================================================
# Module 6 — Models
# ============================================================

@app.get("/algos")
def algos_list():
    return mdl.list_algos()


class ModelCfgReq(BaseModel):
    name: str
    algo: str
    params: dict = {}


@app.post("/models")
def models_create(req: ModelCfgReq):
    if req.algo not in mdl.MODEL_SPECS:
        raise HTTPException(400, f"Unknown algo: {req.algo}")
    mid = su.new_id("m_")
    spec = mdl.MODEL_SPECS[req.algo]
    meta = _common_meta(req.name, "models", id=mid,
                        algo=req.algo, params=req.params,
                        kind_of_algo=spec["kind"], task=spec["task"])
    su.save_meta("models", mid, meta)
    return meta


@app.get("/models")
def models_list():
    return su.list_kind("models")


@app.get("/models/{mid}")
def models_get(mid: str):
    meta = su.get_meta("models", mid)
    if not meta:
        raise HTTPException(404, "model not found")
    return meta


@app.delete("/models/{mid}")
def models_delete(mid: str):
    return {"deleted": su.delete_item("models", mid)}


# ============================================================
# Module 7 — HPO
# ============================================================

class HPOReq(BaseModel):
    name: str
    config: dict


@app.post("/hpo")
def hpo_create(req: HPOReq):
    cid = su.new_id("hpo_")
    meta = _common_meta(req.name, "configs", id=cid,
                        config_kind="hpo", config=req.config)
    su.save_meta("configs", cid, meta)
    return meta


@app.get("/hpo")
def hpo_list():
    return [m for m in su.list_kind("configs")
            if m.get("config_kind") == "hpo"]


@app.get("/hpo/{cid}")
def hpo_get(cid: str):
    meta = su.get_meta("configs", cid)
    if not meta or meta.get("config_kind") != "hpo":
        raise HTTPException(404, "hpo config not found")
    return meta


@app.delete("/hpo/{cid}")
def hpo_delete(cid: str):
    return {"deleted": su.delete_item("configs", cid)}


# ============================================================
# Module 8 — Pipelines
# ============================================================

class PipelineReq(BaseModel):
    name: str
    feature_set_ids: list[str]
    feature_combine: str = "concat"
    feature_compute_id: str | None = None
    feature_select_id: str | None = None
    label_set_id: str
    model_chain: list[str]
    hpo_id: str | None = None
    device: str = "cpu"
    val_ratio: float = 0.2
    convergence: dict = {"early_stop_patience": 30, "min_delta": 1e-5}


@app.post("/pipelines")
def pipelines_create(req: PipelineReq):
    pid = su.new_id("pp_")
    body = req.model_dump()
    name = body.pop("name")
    meta = _common_meta(name, "pipelines", id=pid, **body)
    meta["id"] = pid
    su.save_meta("pipelines", pid, meta)
    return meta


@app.get("/pipelines")
def pipelines_list():
    return su.list_kind("pipelines")


@app.get("/pipelines/{pid}")
def pipelines_get(pid: str):
    meta = su.get_meta("pipelines", pid)
    if not meta:
        raise HTTPException(404, "pipeline not found")
    return meta


@app.delete("/pipelines/{pid}")
def pipelines_delete(pid: str):
    return {"deleted": su.delete_item("pipelines", pid)}


def _run_pipeline_thread(pid: str):
    _RUN_STATUS[pid] = {"status": "running", "events": [], "archive_id": None,
                        "error": None}

    def cb(evt: dict):
        _RUN_STATUS[pid]["events"].append(evt)
        if len(_RUN_STATUS[pid]["events"]) > 1000:
            _RUN_STATUS[pid]["events"] = _RUN_STATUS[pid]["events"][-1000:]

    try:
        archive_id = training.run_pipeline(pid, progress_cb=cb)
        _RUN_STATUS[pid].update({"status": "done", "archive_id": archive_id})
    except Exception as e:
        traceback.print_exc()
        _RUN_STATUS[pid].update({"status": "error", "error": str(e)})


@app.post("/pipelines/{pid}/run")
def pipelines_run(pid: str):
    meta = su.get_meta("pipelines", pid)
    if not meta:
        raise HTTPException(404, "pipeline not found")
    cur = _RUN_STATUS.get(pid)
    if cur and cur.get("status") == "running":
        raise HTTPException(400, "already running")
    t = threading.Thread(target=_run_pipeline_thread, args=(pid,), daemon=True)
    t.start()
    return {"started": True, "pipeline_id": pid}


@app.get("/pipelines/{pid}/run_status")
def pipelines_run_status(pid: str, since: int = 0):
    cur = _RUN_STATUS.get(pid) or {"status": "idle", "events": [],
                                   "archive_id": None}
    events = cur.get("events", [])
    return {
        "status": cur.get("status"),
        "archive_id": cur.get("archive_id"),
        "error": cur.get("error"),
        "n_total_events": len(events),
        "events": events[since:],
        "next_since": len(events),
    }


# ============================================================
# Module 9 — Archives (results & evaluation)
# ============================================================

@app.get("/archives")
def archives_list():
    return su.list_kind("archives")


@app.get("/archives/{aid}")
def archives_get(aid: str):
    meta = su.get_meta("archives", aid)
    if not meta:
        raise HTTPException(404, "archive not found")
    arc_dir = Path(meta["path"])
    files = sorted([f.name for f in arc_dir.iterdir() if f.is_file()])
    extras = {}
    for fname in ("final_metrics.json", "final_confusion_matrix.json",
                  "iteration_log.json", "tp_fp_fn_tn.json", "hpo_result.json",
                  "trial_records.json", "feature_importances.json",
                  "cluster_result.json", "ae_records.json"):
        p = arc_dir / fname
        if p.exists():
            try:
                extras[fname] = su.read_json(p)
            except Exception:
                pass
    return {"meta": meta, "files": files, "extras": extras}


@app.get("/archives/{aid}/file/{name}")
def archives_file(aid: str, name: str):
    meta = su.get_meta("archives", aid)
    if not meta:
        raise HTTPException(404, "archive not found")
    p = Path(meta["path"]) / name
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"file {name} not in archive")
    return FileResponse(p, filename=name)


@app.delete("/archives/{aid}")
def archives_delete(aid: str):
    meta = su.get_meta("archives", aid)
    if not meta:
        return {"deleted": False}
    p = Path(meta["path"])
    if p.exists() and p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    su.delete_item("archives", aid)
    return {"deleted": True}


# ============================================================
# Static frontend
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    p = FRONTEND_DIR / "index.html"
    if not p.exists():
        return HTMLResponse("<h1>Frontend missing</h1>", status_code=500)
    return HTMLResponse(p.read_text(encoding="utf-8"))


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)),
              name="static")
