"""Log every finished run under results/ to an MLflow tracking store and export a runs table.

    MLFLOW_DISABLE_AGENT_HINT=1 PYTHONPATH=src python scripts/log_runs_to_mlflow.py \
        --tracking-uri file:///tmp/smft_mlruns --out results/mlflow_runs.csv

Training writes metrics.json, config.yaml and predictions.npz for every run. This
script turns those into MLflow runs (params, final metrics, the learning curve as
step metrics, and the files as artifacts), then exports the runs table so the
tracking record is visible in the repository.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mlflow


def flatten(cfg: dict) -> dict:
    return {k: str(v) for k, v in cfg.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default="file:///tmp/smft_mlruns")
    ap.add_argument("--experiment", default="bert-mini-sst2")
    ap.add_argument("--out", type=Path, default=Path("results/mlflow_runs.csv"))
    args = ap.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    paths = sorted(Path("results").glob("*/metrics.json")) + sorted(Path("results/runs").glob("*/metrics.json"))
    for p in paths:
        rec = json.loads(p.read_text())
        name = p.parent.name
        group = re.sub(r"_s\d+$", "", name)
        with mlflow.start_run(run_name=name):
            mlflow.log_params(flatten(rec["config"]))
            mlflow.set_tags({"group": group, "results_dir": str(p.parent), "device": rec["hardware"]["device"],
                             "torch": rec["hardware"]["torch"]})
            metrics = {"train_seconds": rec["train_seconds"], "trainable_params": rec["trainable_params"],
                       "best_step": rec["best_step"]}
            for split, m in rec["eval"].items():
                for k in ("accuracy", "f1", "roc_auc", "brier", "ece", "loss"):
                    metrics[f"{split}_{k}"] = m[k]
            mlflow.log_metrics(metrics)
            for point in rec["curve"]:
                mlflow.log_metrics({f"curve_{k}": v for k, v in point.items() if k not in ("step", "epoch")},
                                   step=point["step"])
            for f in ("metrics.json", "config.yaml"):
                mlflow.log_artifact(str(p.parent / f))
    runs = mlflow.search_runs(experiment_names=[args.experiment], order_by=["tags.mlflow.runName ASC"])
    cols = ["run_id", "tags.mlflow.runName", "tags.group"] + sorted(
        c for c in runs.columns if c.startswith("params.") or (c.startswith("metrics.") and not c.startswith("metrics.curve_")))
    runs[cols].to_csv(args.out, index=False)
    print(f"logged {len(paths)} runs, exported {args.out}")


if __name__ == "__main__":
    main()
