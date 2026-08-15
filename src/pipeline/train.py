"""Phase 1.5/1.6 + Phase 5 — train XGBoost, run the leakage ablation, export ONNX, log to MLflow.

Usage:
    python -m src.pipeline.train --data "data/2022*/*.csv" --cutoff 2022-05-01

Requires requirements-train.txt (not part of the serving image).
"""
import argparse
import json
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import average_precision_score, roc_auc_score

from src.pipeline.features import add_labels, grouped_temporal_split, naive_random_split
from src.pipeline.ingest import SMART_COLUMNS, load_quarter

FEATURE_COLUMNS = ["capacity_bytes"] + SMART_COLUMNS


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    top_k = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_k].mean())


def fit_and_evaluate(train: pl.DataFrame, test: pl.DataFrame, run_name: str, split_name: str):
    X_train = train.select(FEATURE_COLUMNS).to_numpy()
    y_train = train["label"].to_numpy()
    X_test = test.select(FEATURE_COLUMNS).to_numpy()
    y_test = test["label"].to_numpy()

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)
    p_at_100 = precision_at_k(y_test, y_score, k=min(100, len(y_test)))

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {"split": split_name, "n_estimators": 400, "max_depth": 6, "scale_pos_weight": spw}
        )
        mlflow.log_metrics({"pr_auc": pr_auc, "roc_auc": roc_auc, "precision_at_100": p_at_100})
        mlflow.xgboost.log_model(model, "model")

    print(f"[{split_name}] pr_auc={pr_auc:.4f} roc_auc={roc_auc:.4f} precision@100={p_at_100:.4f}")
    return model, pr_auc


def export_onnx(model: xgb.XGBClassifier, out_path: str) -> None:
    onnx_model = convert_xgboost(
        model, initial_types=[("input", FloatTensorType([None, len(FEATURE_COLUMNS)]))]
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", required=True, nargs="+", help="One or more glob paths to quarterly CSVs"
    )
    parser.add_argument("--cutoff", required=True, help="YYYY-MM-DD temporal split cutoff")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()

    mlflow.set_experiment("drive-failure")

    raw = load_quarter(args.data)
    labeled = add_labels(raw)

    cutoff = date.fromisoformat(args.cutoff)
    grouped_train, grouped_test = grouped_temporal_split(labeled, cutoff)
    naive_train, naive_test = naive_random_split(labeled)

    fit_and_evaluate(naive_train, naive_test, "xgb-naive-random", "naive_random")
    model, pr_auc = fit_and_evaluate(
        grouped_train, grouped_test, "xgb-grouped-temporal", "grouped_temporal"
    )

    onnx_path = f"{args.artifacts_dir}/model.onnx"
    export_onnx(model, onnx_path)
    model.save_model(f"{args.artifacts_dir}/xgb_model.json")

    config_path = f"{args.artifacts_dir}/feature_config.json"
    config = json.loads(Path(config_path).read_text())
    config["model_version"] = f"xgb-{cutoff.isoformat()}-{pr_auc:.3f}"
    Path(config_path).write_text(json.dumps(config, indent=2))

    print(f"Wrote {onnx_path}, model_version={config['model_version']}")


if __name__ == "__main__":
    main()
