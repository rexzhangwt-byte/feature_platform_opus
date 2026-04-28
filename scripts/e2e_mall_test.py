"""End-to-end test for the mall classification task.

Walks through every module via the REST API:
  1. upload datasets (people_count + dwell_time)
  2. promote them to feature sets
  3. build a feature-compute graph (standardize + concat + filter_keys by labels)
  4. upload labels
  5. create a feature-selection config (variance filter + uniform importance)
  6. create AE model config + MLP model config
  7. create an HPO config (random search on MLP hidden_dim/lr)
  8. create a pipeline (AE -> MLP) using both feature sets
  9. run pipeline, poll progress
 10. fetch archive and assert that TP/FP/FN/TN keys + confusion matrix exist
"""
import json
import time
from pathlib import Path

import requests

BASE = "http://localhost:8765"
DATA = Path(__file__).parent / "mall_dataset"


def post_json(path, body):
    r = requests.post(BASE + path, json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def upload_file(path, name, file_path):
    with open(file_path, "rb") as f:
        r = requests.post(BASE + path, data={"name": name},
                          files={"file": (file_path.name, f, "application/json")},
                          timeout=60)
    r.raise_for_status()
    return r.json()


def run():
    # 1. datasets
    ds_pc = upload_file("/datasets/upload", "people_count_raw",
                        DATA / "people_count.json")
    ds_dw = upload_file("/datasets/upload", "dwell_time_raw",
                        DATA / "dwell_time.json")
    print("datasets:", ds_pc["id"], ds_dw["id"])

    # 2. labels
    lb = upload_file("/labels/upload", "mall_label", DATA / "mall_label.json")
    print("label set:", lb["id"], "dist=", lb["distribution"])

    # 3. promote both datasets to feature sets directly
    fs_pc = post_json("/features/from_dataset",
                      {"dataset_id": ds_pc["id"], "name": "people_count_fs"})
    fs_dw = post_json("/features/from_dataset",
                      {"dataset_id": ds_dw["id"], "name": "dwell_time_fs"})
    print("feature sets (raw):", fs_pc["id"], fs_dw["id"])

    # 4. build a feature-compute graph for people_count:
    #    input -> standardize -> concat(orig, std) -> filter_keys(by label) -> output
    graph_pc = {
        "nodes": [
            {"id": "in_raw", "type": "input",
             "ref_kind": "features", "ref_id": fs_pc["id"],
             "x": 30, "y": 30},
            {"id": "in_raw_b", "type": "input",
             "ref_kind": "features", "ref_id": fs_pc["id"],
             "x": 30, "y": 130},
            {"id": "std", "type": "op", "op_type": "standardize",
             "params": {}, "x": 220, "y": 30},
            {"id": "cat", "type": "op", "op_type": "concat",
             "params": {}, "x": 410, "y": 80},
            {"id": "filt", "type": "op", "op_type": "filter_keys",
             "params": {"allow_from_ref": lb["id"], "allow_ref_kind": "labels"},
             "x": 600, "y": 80},
            {"id": "out", "type": "output",
             "output_name": "people_count_std_cat_filtered",
             "x": 790, "y": 80},
        ],
        "edges": [
            {"from": "in_raw",   "to": "std",  "port": 0},
            {"from": "in_raw_b", "to": "cat",  "port": 0},
            {"from": "std",      "to": "cat",  "port": 1},
            {"from": "cat",      "to": "filt", "port": 0},
            {"from": "filt",     "to": "out",  "port": 0},
        ],
    }
    fc_pc = post_json("/feature_compute",
                      {"name": "pc_std_cat_filter", "graph": graph_pc})
    fc_pc_run = post_json(f"/feature_compute/{fc_pc['id']}/run", {})
    fs_pc_processed = fc_pc_run["feature_sets"][0]
    print("processed people_count fs:", fs_pc_processed["id"],
          "summary=", fs_pc_processed["summary"]["num_keys"], "keys")

    # same for dwell_time
    graph_dw = json.loads(json.dumps(graph_pc))
    for n in graph_dw["nodes"]:
        if n["type"] == "input":
            n["ref_id"] = fs_dw["id"]
        if n["type"] == "output":
            n["output_name"] = "dwell_std_cat_filtered"
    fc_dw = post_json("/feature_compute",
                      {"name": "dw_std_cat_filter", "graph": graph_dw})
    fc_dw_run = post_json(f"/feature_compute/{fc_dw['id']}/run", {})
    fs_dw_processed = fc_dw_run["feature_sets"][0]
    print("processed dwell fs:", fs_dw_processed["id"])

    # 5. feature selection
    fsel = post_json("/feature_select", {
        "name": "low_variance_drop",
        "config": {
            "filters": [
                {"type": "missing_drop"},
                {"type": "variance", "min_var": 1e-6},
            ],
            "importance": {"strategy": "uniform"},
        },
    })
    print("fsel:", fsel["id"])

    # 6. models
    ae = post_json("/models", {
        "name": "AE_32",
        "algo": "autoencoder_torch",
        "params": {"hidden_dim": 64, "latent_dim": 16,
                   "epochs": 60, "lr": 1e-3, "batch_size": 32},
    })
    mlp = post_json("/models", {
        "name": "MLP",
        "algo": "mlp_torch",
        "params": {"hidden_dim": 32, "epochs": 60, "lr": 1e-3,
                   "batch_size": 32, "dropout": 0.0},
    })
    print("AE/MLP:", ae["id"], mlp["id"])

    # 7. HPO
    hpo = post_json("/hpo", {
        "name": "MLP_random",
        "config": {
            "strategy": "random",
            "max_trials": 3,
            "param_space": {
                "hidden_dim": {"type": "int", "low": 16, "high": 64},
                "lr":         {"type": "float", "low": 1e-4, "high": 1e-2,
                               "log": True},
            },
            "priorities": ["hidden_dim", "lr"],
            "seed": 0,
        },
    })
    print("hpo:", hpo["id"])

    # 8. pipeline
    pipe = post_json("/pipelines", {
        "name": "商场分类",
        "feature_set_ids": [fs_pc_processed["id"], fs_dw_processed["id"]],
        "feature_combine": "concat",
        "feature_compute_id": fc_pc["id"],
        "feature_select_id": fsel["id"],
        "label_set_id": lb["id"],
        "model_chain": [ae["id"], mlp["id"]],
        "hpo_id": hpo["id"],
        "device": "cpu",
        "val_ratio": 0.2,
        "convergence": {"early_stop_patience": 8, "min_delta": 1e-4},
    })
    print("pipeline:", pipe["id"])

    # 9. run
    post_json(f"/pipelines/{pipe['id']}/run", {})
    since = 0
    while True:
        r = requests.get(f"{BASE}/pipelines/{pipe['id']}/run_status?since={since}",
                         timeout=10).json()
        for ev in r["events"]:
            print(f"  [{ev['phase']}] {ev['msg']}")
        since = r["next_since"]
        if r["status"] == "done":
            archive_id = r["archive_id"]
            break
        if r["status"] == "error":
            raise RuntimeError(f"Pipeline failed: {r['error']}")
        time.sleep(0.5)
    print("archive_id:", archive_id)

    # 10. archive details
    detail = requests.get(f"{BASE}/archives/{archive_id}", timeout=30).json()
    print("\n==== Final metrics ====")
    print(json.dumps(detail["extras"]["final_metrics.json"], indent=2,
                     ensure_ascii=False))
    cm = detail["extras"]["final_confusion_matrix.json"]
    print("Confusion matrix:")
    print("  labels:", cm["labels"])
    print("  matrix:", cm["matrix"])
    print(f"  accuracy={cm['accuracy']:.3f} f1_macro={cm['f1_macro']:.3f}")
    tp = detail["extras"]["tp_fp_fn_tn.json"]
    print("TP/FP/FN/TN counts:")
    for k in ["TP", "FP", "FN", "TN"]:
        print(f"  {k}: {len(tp[k])} (e.g. {tp[k][:3]})")
    assert "iteration_log.json" in detail["extras"]
    assert detail["extras"]["hpo_result.json"]["n_trials"] == 3
    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    run()
