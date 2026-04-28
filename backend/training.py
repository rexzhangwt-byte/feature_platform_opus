"""Training pipeline (Module 7) + archive/evaluation (Module 8).

A pipeline (model training task) ties together:
  - feature_set_ids        : list[str]   -> we take SAMPLE INTERSECTION across them
  - feature_combine        : "concat" | "sum" | "mean"
  - feature_select_id      : str|None
  - label_set_id           : str
  - model_chain            : [model_config_id, ...]   (e.g. [AE_id, MLP_id])
  - hpo_id                 : str|None    (applied to the LAST model in chain)
  - device                 : "cpu"|"gpu"
  - convergence            : {"early_stop_patience": int, "min_delta": float}
  - val_ratio              : float (e.g. 0.2)

Outputs an archive containing:
  - best params per model
  - best weights for torch models / fitted estimator pickle for sklearn
  - confusion matrix per HPO trial AND for the best model
  - F1 / accuracy / precision / recall
  - Validation set TP / FP / FN / TN keys (json) for human review
  - Iteration history with reasons (Module 7 ask)
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from . import data_io
from . import feature_compute
from . import feature_select as fsel
from . import models as mdl
from . import hpo as hpo_mod
from . import storage_utils as su


# ------------------------------------------------------------
# Feature assembly (multi-feature-set intersection + combine)
# ------------------------------------------------------------

def assemble_features(feature_set_ids: list[str],
                      combine: str,
                      label_keys: list[str] | None = None
                      ) -> tuple[np.ndarray, list[str]]:
    """Load each feature set, intersect on keys (and label keys if given),
    then combine per the strategy."""
    if not feature_set_ids:
        raise ValueError("No feature sets selected")
    sets = []
    for fid in feature_set_ids:
        meta = su.get_meta("features", fid)
        if not meta:
            raise ValueError(f"Feature set not found: {fid}")
        d = data_io.load_dict_file(meta["path"])
        sets.append(d)

    common = set(sets[0].keys())
    for s in sets[1:]:
        common &= set(s.keys())
    if label_keys is not None:
        common &= set(label_keys)
    keys = sorted(common)

    if not keys:
        raise ValueError("No overlapping sample keys across selected feature sets / labels")

    matrices = []
    for s in sets:
        m, _ = data_io.to_matrix(s, keys)
        matrices.append(m)

    if combine == "concat":
        X = np.concatenate(matrices, axis=1)
    else:
        L = min(m.shape[1] for m in matrices)
        matrices = [m[:, :L] for m in matrices]
        stacked = np.stack(matrices, axis=0)
        if combine == "sum":
            X = stacked.sum(axis=0)
        elif combine == "mean":
            X = stacked.mean(axis=0)
        else:
            raise ValueError(f"Unknown combine strategy: {combine}")
    return X, keys


# ------------------------------------------------------------
# Torch helpers
# ------------------------------------------------------------

def _device_from_pref(pref: str) -> torch.device:
    if pref == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_autoencoder(X: np.ndarray, params: dict, device: torch.device,
                      patience: int = 30, min_delta: float = 1e-5,
                      log_cb: Callable[[str], None] | None = None
                      ) -> tuple[mdl.TorchAE, np.ndarray, list[dict]]:
    input_dim = X.shape[1]
    hidden = int(params.get("hidden_dim", 64))
    latent = int(params.get("latent_dim", 32))
    epochs = int(params.get("epochs", 600))
    lr = float(params.get("lr", 1e-3))
    bs = int(params.get("batch_size", 64))

    model = mdl.TorchAE(input_dim, hidden, latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    n = Xt.shape[0]
    log = []
    best_loss = float("inf")
    bad = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xt[idx]
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * xb.shape[0]
        avg = total / max(n, 1)
        log.append({"epoch": ep, "loss": avg})
        if avg + min_delta < best_loss:
            best_loss = avg
            bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if log_cb:
                    log_cb(f"AE early stop at epoch {ep+1} (best loss {best_loss:.6f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, Z = model(Xt)
    return model, Z.cpu().numpy(), log


def train_mlp_torch(X: np.ndarray, y: np.ndarray, params: dict,
                    device: torch.device, val_ratio: float,
                    patience: int = 30, min_delta: float = 1e-5,
                    log_cb: Callable[[str], None] | None = None,
                    seed: int = 42
                    ) -> tuple[mdl.TorchMLP, dict, list[dict]]:
    classes = sorted(set(y.tolist()))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.asarray([cls_to_idx[c] for c in y], dtype=np.int64)

    X_tr, X_va, y_tr, y_va, idx_tr, idx_va = train_test_split(
        X, y_idx, np.arange(len(y_idx)),
        test_size=max(0.05, min(0.5, val_ratio)),
        random_state=seed,
        stratify=y_idx if len(classes) > 1 and min(np.bincount(y_idx)) >= 2 else None,
    )

    input_dim = X.shape[1]
    hidden = int(params.get("hidden_dim", 64))
    epochs = int(params.get("epochs", 600))
    lr = float(params.get("lr", 1e-3))
    bs = int(params.get("batch_size", 64))
    dropout = float(params.get("dropout", 0.0))

    model = mdl.TorchMLP(input_dim, hidden, len(classes), dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Xt_tr = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    yt_tr = torch.from_numpy(y_tr).to(device)
    Xt_va = torch.from_numpy(X_va.astype(np.float32)).to(device)
    yt_va = torch.from_numpy(y_va).to(device)

    n = Xt_tr.shape[0]
    log = []
    best_acc = -1.0
    bad = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        model.train()
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = model(Xt_tr[idx])
            loss = loss_fn(logits, yt_tr[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * idx.shape[0]
        train_loss = total / max(n, 1)

        model.eval()
        with torch.no_grad():
            preds_va = model(Xt_va).argmax(dim=1).cpu().numpy()
            preds_tr = model(Xt_tr).argmax(dim=1).cpu().numpy()
        acc_va = float((preds_va == y_va).mean())
        acc_tr = float((preds_tr == y_tr).mean())
        log.append({"epoch": ep, "train_loss": train_loss,
                    "val_acc": acc_va, "train_acc": acc_tr})

        if acc_va > best_acc + min_delta:
            best_acc = acc_va
            bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if log_cb:
                    log_cb(f"MLP early stop at epoch {ep+1} (best val_acc {best_acc:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_va = model(Xt_va).argmax(dim=1).cpu().numpy()
        preds_tr = model(Xt_tr).argmax(dim=1).cpu().numpy()

    splits = {
        "train_idx": idx_tr.tolist(), "val_idx": idx_va.tolist(),
        "y_train": y_tr.tolist(), "y_val": y_va.tolist(),
        "preds_train": preds_tr.tolist(), "preds_val": preds_va.tolist(),
        "classes": classes, "best_val_acc": best_acc,
    }
    return model, splits, log


# ------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------

def confusion_matrix_payload(y_true: list[int], y_pred: list[int],
                             classes: list[Any]) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    return {
        "labels": [str(c) for c in classes],
        "matrix": cm.tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
    }


def split_tp_fp_fn_tn(keys: list[str], y_true: list[int],
                      y_pred: list[int], classes: list[Any],
                      positive_class: Any | None = None) -> dict:
    """Binary-style split. If multi-class, computes one-vs-rest for the
    selected positive class (default = mall/positive heuristic)."""
    if not classes:
        return {"TP": [], "FP": [], "FN": [], "TN": []}
    if positive_class is None:
        for c in classes:
            if str(c) in ("商场", "mall", "1", "True", "true", "Y"):
                positive_class = c
                break
        if positive_class is None:
            positive_class = classes[1] if len(classes) > 1 else classes[0]

    pos_idx = classes.index(positive_class)
    out = {"TP": [], "FP": [], "FN": [], "TN": [],
           "positive_class": str(positive_class)}
    for k, yt, yp in zip(keys, y_true, y_pred):
        is_pos_true = (yt == pos_idx)
        is_pos_pred = (yp == pos_idx)
        if is_pos_true and is_pos_pred:
            out["TP"].append(k)
        elif (not is_pos_true) and is_pos_pred:
            out["FP"].append(k)
        elif is_pos_true and (not is_pos_pred):
            out["FN"].append(k)
        else:
            out["TN"].append(k)
    return out


# ------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------

def _resolve_input_for_compute(ref_kind: str, ref_id: str) -> dict:
    meta = su.get_meta(ref_kind, ref_id)
    if not meta:
        raise ValueError(f"{ref_kind}/{ref_id} not found")
    return data_io.load_dict_file(meta["path"])


def materialize_compute_config(compute_cfg_id: str) -> dict[str, dict]:
    cfg_meta = su.get_meta("configs", compute_cfg_id)
    if not cfg_meta or cfg_meta.get("config_kind") != "feature_compute":
        raise ValueError("Bad feature_compute config id")
    graph = cfg_meta["graph"]
    res = feature_compute.execute_graph(graph, _resolve_input_for_compute)
    return res["_outputs"]


def run_pipeline(pipeline_id: str,
                 progress_cb: Callable[[dict], None] | None = None) -> str:
    p_meta = su.get_meta("pipelines", pipeline_id)
    if not p_meta:
        raise ValueError("pipeline not found")

    def emit(phase: str, msg: str, data: dict | None = None):
        if progress_cb:
            progress_cb({"phase": phase, "msg": msg, "data": data or {},
                         "ts": time.time()})

    iter_log: list[dict] = []
    def log_reason(reason: str, **extra):
        iter_log.append({"ts": time.time(), "reason": reason, **extra})
        emit("log", reason, extra)

    # --- 1. Load labels ----
    label_id = p_meta["label_set_id"]
    label_meta = su.get_meta("labels", label_id)
    labels = data_io.load_dict_file(label_meta["path"])
    label_keys = list(labels.keys())
    log_reason(f"Loaded label set '{label_meta.get('name')}' with "
               f"{len(label_keys)} samples")

    # --- 2. Assemble features (intersection across feature sets + labels) ----
    fset_ids = p_meta["feature_set_ids"]
    combine = p_meta.get("feature_combine", "concat")
    X, keys = assemble_features(fset_ids, combine, label_keys=label_keys)
    log_reason(
        f"Assembled features: intersection of {len(fset_ids)} feature sets "
        f"+ labels => {X.shape[0]} samples × {X.shape[1]} dims (combine={combine})"
    )

    # --- 3. Feature selection ----
    fs_id = p_meta.get("feature_select_id")
    if fs_id:
        fs_meta = su.get_meta("configs", fs_id)
        cfg = fs_meta["config"]
        X, keys, _, fs_report = fsel.apply_filters(X, keys, cfg.get("filters", []))
        X, weights = fsel.apply_importance(X, cfg.get("importance", {}))
        log_reason(
            f"Feature selection '{fs_meta.get('name')}' applied. "
            f"Filters report: {fs_report['steps']}; "
            f"importance strategy={cfg.get('importance', {}).get('strategy', 'uniform')}"
        )
    else:
        log_reason("No feature selection config — using all features as-is")

    # --- 4. Build label vector aligned to keys ----
    y_raw = [labels[k] for k in keys]
    classes = sorted({str(v) for v in y_raw})
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.asarray([cls_to_idx[str(v)] for v in y_raw], dtype=np.int64)
    log_reason(f"Label classes = {classes}; class counts = "
               f"{dict(zip(classes, np.bincount(y).tolist()))}")

    # --- 5. Resolve model chain ----
    model_chain = p_meta.get("model_chain", [])
    if not model_chain:
        raise ValueError("Pipeline has no model_chain")
    model_metas = [su.get_meta("models", mid) for mid in model_chain]
    if any(m is None for m in model_metas):
        raise ValueError("Some model config in chain not found")

    device = _device_from_pref(p_meta.get("device", "cpu"))
    log_reason(f"Compute device = {device}")
    val_ratio = float(p_meta.get("val_ratio", 0.2))
    conv = p_meta.get("convergence") or {}
    patience = int(conv.get("early_stop_patience", 30))
    min_delta = float(conv.get("min_delta", 1e-5))

    # --- 6. AE preprocessing (apply chain in order, except last) ----
    X_cur = X.astype(np.float32)
    ae_records = []
    for m_meta in model_metas[:-1]:
        algo = m_meta["algo"]
        params = m_meta.get("params", {}) or {}
        if algo == "autoencoder_torch":
            log_reason(
                f"Training AE '{m_meta.get('name')}' "
                f"(input_dim={X_cur.shape[1]}, latent_dim="
                f"{params.get('latent_dim', 32)}, epochs={params.get('epochs', 600)})"
            )
            ae_model, Z, ae_log = train_autoencoder(
                X_cur, params, device, patience, min_delta, log_cb=log_reason
            )
            ae_records.append({
                "model_id": m_meta["id"],
                "algo": algo,
                "epochs_run": len(ae_log),
                "final_loss": ae_log[-1]["loss"] if ae_log else None,
                "state_keys": list(ae_model.state_dict().keys()),
            })
            X_cur = Z
            log_reason(f"AE encoded features → shape {X_cur.shape}")
        else:
            log_reason(f"Pre-stage model '{algo}' is not AE — skipping (only AE preprocessing supported in chain)")

    # --- 7. Final model + HPO ----
    final_meta = model_metas[-1]
    final_algo = final_meta["algo"]
    final_params = dict(final_meta.get("params", {}) or {})
    log_reason(f"Final model = {final_algo} with base params {final_params}")

    hpo_id = p_meta.get("hpo_id")
    archive_id = su.new_id("archive_")
    archive_dir = su.kind_dir("archives") / archive_id
    archive_dir.mkdir(parents=True, exist_ok=True)

    trial_records: list[dict] = []

    def objective(trial_params: dict) -> float:
        merged = {**final_params, **trial_params}
        score, payload = _train_and_score_final(
            final_algo, merged, X_cur, y, classes, keys,
            device, val_ratio, patience, min_delta, log_reason
        )
        trial_records.append({"params": merged, "score": score,
                              "metrics": payload["metrics"],
                              "confusion_matrix": payload["confusion_matrix"]})
        return score

    if hpo_id:
        hpo_meta = su.get_meta("configs", hpo_id)
        hcfg = hpo_meta["config"]
        log_reason(f"Starting HPO '{hpo_meta.get('name')}' strategy="
                   f"{hcfg.get('strategy')} max_trials={hcfg.get('max_trials')}")
        result = hpo_mod.run_search(
            strategy=hcfg.get("strategy", "random"),
            param_space=hcfg.get("param_space", {}),
            objective=objective,
            max_trials=int(hcfg.get("max_trials", 10)),
            priorities=hcfg.get("priorities"),
            bayes_cfg=hcfg.get("bayes"),
            seed=int(hcfg.get("seed", 0)),
            on_trial=lambda i, p, s, r: log_reason(
                f"HPO trial {i+1}: params={p} score={s:.4f} ({r})"),
        )
        best_params = result["best"]["params"] or {}
        merged = {**final_params, **best_params}
        log_reason(f"HPO finished. Best score={result['best']['score']:.4f} "
                   f"params={best_params}")
    else:
        result = {"best": {"score": None, "params": final_params, "idx": 0},
                  "history": [], "strategy": "none", "n_trials": 1}
        merged = final_params
        log_reason("No HPO — single training run with base params")

    # --- 8. Final fit with best params, full archive ----
    final_score, final_payload = _train_and_score_final(
        final_algo, merged, X_cur, y, classes, keys,
        device, val_ratio, patience, min_delta, log_reason,
        save_dir=archive_dir,
    )
    log_reason(f"Final model score = {final_score:.4f}")

    # save iteration log + everything
    su.write_json(archive_dir / "iteration_log.json", iter_log)
    su.write_json(archive_dir / "hpo_result.json", result)
    su.write_json(archive_dir / "trial_records.json", trial_records)
    su.write_json(archive_dir / "ae_records.json", ae_records)
    su.write_json(archive_dir / "final_metrics.json", final_payload["metrics"])
    su.write_json(archive_dir / "final_confusion_matrix.json",
                  final_payload["confusion_matrix"])
    su.write_json(archive_dir / "val_split_keys.json", final_payload["val_split"])
    su.write_json(archive_dir / "tp_fp_fn_tn.json", final_payload["tp_fp_fn_tn"])
    if "feature_importances" in final_payload:
        su.write_json(archive_dir / "feature_importances.json",
                      final_payload["feature_importances"])
    if "cluster_result" in final_payload:
        su.write_json(archive_dir / "cluster_result.json",
                      final_payload["cluster_result"])

    archive_meta = {
        "id": archive_id,
        "kind": "archives",
        "name": f"{p_meta.get('name','pipeline')}_run_{int(time.time())}",
        "pipeline_id": pipeline_id,
        "pipeline_name": p_meta.get("name"),
        "created_at": su.now_ts(),
        "path": str(archive_dir),
        "device": str(device),
        "feature_shape_after_ae": list(X_cur.shape),
        "n_classes": len(classes),
        "classes": classes,
        "best_params": merged,
        "best_score": float(final_score),
        "final_metrics": final_payload["metrics"],
        "n_trials": result.get("n_trials", 0),
        "ae_used": [r["model_id"] for r in ae_records],
        "final_algo": final_algo,
    }
    su.save_meta("archives", archive_id, archive_meta)
    emit("done", f"Archive saved: {archive_id}", {"archive_id": archive_id})
    return archive_id


def _train_and_score_final(algo: str, params: dict,
                           X: np.ndarray, y: np.ndarray,
                           classes: list[str], keys: list[str],
                           device: torch.device, val_ratio: float,
                           patience: int, min_delta: float,
                           log_cb: Callable[[str], None],
                           save_dir: Path | None = None
                           ) -> tuple[float, dict]:
    spec = mdl.MODEL_SPECS[algo]
    payload: dict[str, Any] = {}

    if spec["task"] == "clustering":
        km, _ = mdl.build_model(algo, params)
        km.fit(X)
        cluster_ids = km.labels_.tolist()
        cluster_to_label: dict[int, int] = {}
        for c in set(cluster_ids):
            members = [y[i] for i, ci in enumerate(cluster_ids) if ci == c]
            if members:
                cluster_to_label[c] = int(np.bincount(members).argmax())
        preds = [cluster_to_label.get(c, 0) for c in cluster_ids]
        cm = confusion_matrix_payload(y.tolist(), preds, classes)
        score = cm["accuracy"]
        payload["metrics"] = cm
        payload["confusion_matrix"] = cm
        payload["cluster_result"] = {
            "labels": cluster_ids, "keys": keys,
            "cluster_to_label": cluster_to_label,
            "inertia": float(km.inertia_),
        }
        payload["val_split"] = {"train_keys": [], "val_keys": keys,
                                "train_idx": [], "val_idx": list(range(len(keys)))}
        payload["tp_fp_fn_tn"] = split_tp_fp_fn_tn(keys, y.tolist(), preds, classes)
        if save_dir is not None:
            with open(save_dir / "model.pkl", "wb") as f:
                pickle.dump(km, f)
        return score, payload

    if algo == "mlp_torch":
        model, splits, log = train_mlp_torch(
            X, y, params, device, val_ratio, patience, min_delta, log_cb,
        )
        cls = splits["classes"]
        cm = confusion_matrix_payload(splits["y_val"], splits["preds_val"],
                                      [classes[i] for i in cls])
        train_cm = confusion_matrix_payload(splits["y_train"], splits["preds_train"],
                                            [classes[i] for i in cls])
        score = float(splits["best_val_acc"])
        val_keys = [keys[i] for i in splits["val_idx"]]
        train_keys = [keys[i] for i in splits["train_idx"]]
        payload["metrics"] = {"train": train_cm, "val": cm,
                              "best_val_acc": score, "epochs_run": len(log)}
        payload["confusion_matrix"] = cm
        payload["val_split"] = {"train_keys": train_keys, "val_keys": val_keys,
                                "train_idx": splits["train_idx"],
                                "val_idx": splits["val_idx"]}
        payload["tp_fp_fn_tn"] = split_tp_fp_fn_tn(
            val_keys, splits["y_val"], splits["preds_val"],
            [classes[i] for i in cls],
        )
        if save_dir is not None:
            torch.save(model.state_dict(), save_dir / "mlp_state_dict.pt")
            su.write_json(save_dir / "training_log.json", log)
            su.write_json(save_dir / "mlp_arch.json", {
                "input_dim": X.shape[1], "hidden_dim": int(params.get("hidden_dim", 64)),
                "num_classes": len(cls),
                "dropout": float(params.get("dropout", 0.0)),
            })
        return score, payload

    # Generic sklearn classification path
    X_tr, X_va, y_tr, y_va, idx_tr, idx_va = train_test_split(
        X, y, np.arange(len(y)),
        test_size=max(0.05, min(0.5, val_ratio)),
        random_state=int(params.get("random_state", 42)),
        stratify=y if min(np.bincount(y)) >= 2 else None,
    )
    est, _ = mdl.build_model(algo, params)
    est.fit(X_tr, y_tr)
    preds_va = est.predict(X_va).tolist()
    preds_tr = est.predict(X_tr).tolist()
    cm = confusion_matrix_payload(y_va.tolist(), preds_va, classes)
    train_cm = confusion_matrix_payload(y_tr.tolist(), preds_tr, classes)
    score = cm["accuracy"]
    val_keys = [keys[i] for i in idx_va.tolist()]
    train_keys = [keys[i] for i in idx_tr.tolist()]
    payload["metrics"] = {"train": train_cm, "val": cm}
    payload["confusion_matrix"] = cm
    payload["val_split"] = {"train_keys": train_keys, "val_keys": val_keys,
                            "train_idx": idx_tr.tolist(),
                            "val_idx": idx_va.tolist()}
    payload["tp_fp_fn_tn"] = split_tp_fp_fn_tn(val_keys, y_va.tolist(),
                                               preds_va, classes)
    if hasattr(est, "feature_importances_"):
        payload["feature_importances"] = est.feature_importances_.tolist()
    elif hasattr(est, "coef_"):
        payload["feature_importances"] = np.asarray(est.coef_).ravel().tolist()
    if save_dir is not None:
        with open(save_dir / "model.pkl", "wb") as f:
            pickle.dump(est, f)
    return score, payload
