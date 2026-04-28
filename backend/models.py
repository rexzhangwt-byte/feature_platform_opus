"""Model algorithm registry (Module 5).

Each model spec defines:
  - param_schema: list of params with type/range/default
  - factory:      function building a fresh estimator from concrete params
  - kind:         "sklearn" | "torch_ae" | "torch_mlp"

Supported algorithms:
  random_forest, decision_tree, logistic_regression, linear_regression,
  knn, naive_bayes, kmeans, mlp_torch, autoencoder_torch
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ---------- sklearn factories ----------
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans


# ---------- torch wrappers ----------
import torch
import torch.nn as nn


class TorchAE(nn.Module):
    """Simple symmetric autoencoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_classes: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------- Model specs ----------
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "kind": "sklearn",
        "task": "classification",
        "param_schema": [
            {"name": "n_estimators", "type": "int", "default": 100, "range": [10, 500]},
            {"name": "max_depth", "type": "int", "default": 10, "range": [2, 50]},
            {"name": "min_samples_split", "type": "int", "default": 2, "range": [2, 20]},
            {"name": "random_state", "type": "int", "default": 42, "range": [0, 9999]},
        ],
        "factory": lambda p: RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=int(p.get("max_depth", 10)),
            min_samples_split=int(p.get("min_samples_split", 2)),
            random_state=int(p.get("random_state", 42)),
            n_jobs=1,
        ),
    },
    "decision_tree": {
        "kind": "sklearn",
        "task": "classification",
        "param_schema": [
            {"name": "max_depth", "type": "int", "default": 10, "range": [2, 50]},
            {"name": "min_samples_split", "type": "int", "default": 2, "range": [2, 20]},
            {"name": "random_state", "type": "int", "default": 42, "range": [0, 9999]},
        ],
        "factory": lambda p: DecisionTreeClassifier(
            max_depth=int(p.get("max_depth", 10)),
            min_samples_split=int(p.get("min_samples_split", 2)),
            random_state=int(p.get("random_state", 42)),
        ),
    },
    "logistic_regression": {
        "kind": "sklearn",
        "task": "classification",
        "param_schema": [
            {"name": "C", "type": "float", "default": 1.0, "range": [0.001, 100.0], "log": True},
            {"name": "max_iter", "type": "int", "default": 200, "range": [50, 2000]},
        ],
        "factory": lambda p: LogisticRegression(
            C=float(p.get("C", 1.0)),
            max_iter=int(p.get("max_iter", 200)),
        ),
    },
    "linear_regression": {
        "kind": "sklearn",
        "task": "regression",
        "param_schema": [],
        "factory": lambda p: LinearRegression(),
    },
    "knn": {
        "kind": "sklearn",
        "task": "classification",
        "param_schema": [
            {"name": "n_neighbors", "type": "int", "default": 5, "range": [1, 50]},
            {"name": "weights", "type": "choice", "default": "uniform",
             "choices": ["uniform", "distance"]},
        ],
        "factory": lambda p: KNeighborsClassifier(
            n_neighbors=int(p.get("n_neighbors", 5)),
            weights=p.get("weights", "uniform"),
        ),
    },
    "naive_bayes": {
        "kind": "sklearn",
        "task": "classification",
        "param_schema": [
            {"name": "var_smoothing", "type": "float", "default": 1e-9,
             "range": [1e-12, 1e-3], "log": True},
        ],
        "factory": lambda p: GaussianNB(
            var_smoothing=float(p.get("var_smoothing", 1e-9)),
        ),
    },
    "kmeans": {
        "kind": "sklearn",
        "task": "clustering",
        "param_schema": [
            {"name": "n_clusters", "type": "int", "default": 2, "range": [2, 20]},
            {"name": "random_state", "type": "int", "default": 42, "range": [0, 9999]},
            {"name": "n_init", "type": "int", "default": 10, "range": [1, 50]},
        ],
        "factory": lambda p: KMeans(
            n_clusters=int(p.get("n_clusters", 2)),
            random_state=int(p.get("random_state", 42)),
            n_init=int(p.get("n_init", 10)),
        ),
    },
    "autoencoder_torch": {
        "kind": "torch_ae",
        "task": "encoding",
        "param_schema": [
            {"name": "hidden_dim", "type": "int", "default": 64, "range": [8, 512]},
            {"name": "latent_dim", "type": "int", "default": 32, "range": [2, 256]},
            {"name": "epochs", "type": "int", "default": 600, "range": [10, 2000]},
            {"name": "lr", "type": "float", "default": 1e-3, "range": [1e-5, 1e-1], "log": True},
            {"name": "batch_size", "type": "int", "default": 64, "range": [8, 1024]},
        ],
        "factory": lambda p: {"params": dict(p)},  # actual model built at fit-time once input_dim is known
    },
    "mlp_torch": {
        "kind": "torch_mlp",
        "task": "classification",
        "param_schema": [
            {"name": "hidden_dim", "type": "int", "default": 64, "range": [8, 512]},
            {"name": "epochs", "type": "int", "default": 600, "range": [10, 2000]},
            {"name": "lr", "type": "float", "default": 1e-3, "range": [1e-5, 1e-1], "log": True},
            {"name": "batch_size", "type": "int", "default": 64, "range": [8, 1024]},
            {"name": "dropout", "type": "float", "default": 0.0, "range": [0.0, 0.7]},
        ],
        "factory": lambda p: {"params": dict(p)},
    },
}


def build_model(algo: str, params: dict):
    spec = MODEL_SPECS.get(algo)
    if not spec:
        raise ValueError(f"Unknown algorithm: {algo}")
    return spec["factory"](params or {}), spec


def list_algos() -> list[dict]:
    out = []
    for name, spec in MODEL_SPECS.items():
        out.append({
            "name": name,
            "kind": spec["kind"],
            "task": spec["task"],
            "param_schema": spec["param_schema"],
        })
    return out
