"""Hyper-parameter optimization (Module 6).

Search strategies:
  - grid:    cartesian product of explicit grids per param
  - random:  uniform sampling N times in declared param ranges
  - bayes:   skopt Bayesian optimization

The HPO config schema:
{
  "strategy": "grid" | "random" | "bayes",
  "max_trials": int,
  "param_space": {
     "<param_name>": {"type": "int"|"float"|"choice",
                      "low": ..., "high": ..., "log": bool, "values": [...],
                      "grid": [...]}
  },
  "priorities": [param_name, ...],
  "bayes": {"n_initial_points": 5, "acq_func": "EI"}
}
"""
from __future__ import annotations

import itertools
import random
from typing import Any, Callable

import numpy as np


def _grid_points(p_cfg: dict, default_n: int = 5) -> list:
    if "grid" in p_cfg and p_cfg["grid"]:
        return list(p_cfg["grid"])
    t = p_cfg.get("type", "float")
    if t == "choice":
        return list(p_cfg.get("values", []))
    lo = p_cfg["low"]
    hi = p_cfg["high"]
    n = int(p_cfg.get("n", default_n))
    if t == "int":
        if n <= 1:
            return [int(lo)]
        return list(np.linspace(lo, hi, n).round().astype(int).tolist())
    if p_cfg.get("log"):
        return list(np.geomspace(max(lo, 1e-12), hi, n).tolist())
    return list(np.linspace(lo, hi, n).tolist())


def _sample_random(p_cfg: dict, rng: random.Random) -> Any:
    t = p_cfg.get("type", "float")
    if t == "choice":
        return rng.choice(p_cfg.get("values", []))
    lo = p_cfg["low"]
    hi = p_cfg["high"]
    if t == "int":
        return rng.randint(int(lo), int(hi))
    if p_cfg.get("log"):
        a, b = np.log(max(lo, 1e-12)), np.log(hi)
        return float(np.exp(rng.uniform(a, b)))
    return rng.uniform(float(lo), float(hi))


def enumerate_grid(param_space: dict, priorities: list[str] | None = None) -> list[dict]:
    names = list(param_space.keys())
    if priorities:
        rest = [n for n in names if n not in priorities]
        names = [n for n in priorities if n in param_space] + rest
    grids = [_grid_points(param_space[n]) for n in names]
    combos = []
    for vals in itertools.product(*grids):
        combos.append({n: v for n, v in zip(names, vals)})
    return combos


def random_search(param_space: dict, n_trials: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n_trials):
        out.append({n: _sample_random(c, rng) for n, c in param_space.items()})
    return out


def run_search(strategy: str,
               param_space: dict,
               objective: Callable[[dict], float],
               max_trials: int = 20,
               priorities: list[str] | None = None,
               bayes_cfg: dict | None = None,
               seed: int = 0,
               on_trial: Callable[[int, dict, float, str], None] | None = None
               ) -> dict:
    history: list[dict] = []
    best = {"score": -float("inf"), "params": None, "idx": -1}

    def record(idx: int, params: dict, score: float, reason: str):
        history.append({"idx": idx, "params": params,
                        "score": float(score), "reason": reason})
        if score > best["score"]:
            best.update({"score": float(score), "params": dict(params), "idx": idx})
        if on_trial:
            try:
                on_trial(idx, params, float(score), reason)
            except Exception:
                pass

    if strategy == "grid":
        combos = enumerate_grid(param_space, priorities)
        if max_trials and len(combos) > max_trials:
            combos = combos[:max_trials]
        for i, params in enumerate(combos):
            score = objective(params)
            record(i, params, score, f"grid trial {i+1}/{len(combos)}")

    elif strategy == "random":
        combos = random_search(param_space, max_trials, seed=seed)
        for i, params in enumerate(combos):
            score = objective(params)
            record(i, params, score, f"random sample {i+1}/{len(combos)}")

    elif strategy == "bayes":
        try:
            from skopt import Optimizer
            from skopt.space import Real, Integer, Categorical
        except Exception as e:
            raise RuntimeError(f"Bayesian search requires scikit-optimize: {e}")

        names = list(param_space.keys())
        dims = []
        for n in names:
            cfg = param_space[n]
            t = cfg.get("type", "float")
            if t == "int":
                dims.append(Integer(int(cfg["low"]), int(cfg["high"]), name=n))
            elif t == "choice":
                dims.append(Categorical(cfg["values"], name=n))
            else:
                prior = "log-uniform" if cfg.get("log") else "uniform"
                dims.append(Real(float(cfg["low"]), float(cfg["high"]),
                                 prior=prior, name=n))
        bayes_cfg = bayes_cfg or {}
        opt = Optimizer(
            dimensions=dims,
            base_estimator=bayes_cfg.get("base_estimator", "GP"),
            n_initial_points=int(bayes_cfg.get("n_initial_points", 5)),
            acq_func=bayes_cfg.get("acq_func", "EI"),
            random_state=seed,
        )
        for i in range(int(max_trials or 20)):
            x = opt.ask()
            params = {n: (int(v) if isinstance(v, np.integer) else v) for n, v in zip(names, x)}
            score = objective(params)
            opt.tell(x, -float(score))
            record(i, params, score,
                   f"bayes trial {i+1} (acq={bayes_cfg.get('acq_func','EI')}, "
                   f"so_far_best={best['score']:.4f})")

    else:
        raise ValueError(f"Unknown HPO strategy: {strategy}")

    return {"best": best, "history": history,
            "strategy": strategy, "n_trials": len(history)}
