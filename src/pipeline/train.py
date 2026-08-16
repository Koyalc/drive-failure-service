"""Phase 1.5/1.6 + Phase 5 — train XGBoost, run the leakage ablation, export ONNX, log to MLflow.

Usage:
    python -m src.pipeline.train --data "data/Q1_2022/*.csv" "data/Q2_2022/*.csv" --cutoff 2022-05-01

Ingest + labeling + delta-feature computation (the slow part, ~20+ min over two full
quarters) is cached to --cache after the first run. Pass --rebuild-cache to force a
fresh load, e.g. after changing which SMART attributes or delta windows are used.

Requires requirements-train.txt (not part of the serving image).
"""
import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import average_precision_score, roc_auc_score

from src.pipeline.features import (
    add_delta_features,
    add_labels,
    delta7d_column,
    grouped_temporal_split,
    naive_random_split,
)
from src.pipeline.ingest import SMART_ATTRIBUTES, SMART_COLUMNS, load_quarter

FEATURE_COLUMNS = (
    ["capacity_bytes"] + SMART_COLUMNS + [delta7d_column(n) for n in SMART_ATTRIBUTES]
)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    top_k = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_k].mean())


def fit_and_evaluate(
    train: pl.DataFrame,
    test: pl.DataFrame,
    run_name: str,
    split_name: str,
    max_depth: int,
    n_estimators: int,
    learning_rate: float,
    max_delta_step: int,
    early_stopping_tail_days: int | None,
):
    """Fit on `train`, evaluate on `test`. If early_stopping_tail_days is set, carve that many
    trailing days off `train` itself as an eval set for early stopping -- `test` stays untouched.
    """
    fit_rows, eval_set = train, None
    if early_stopping_tail_days:
        val_cutoff = train["date"].max() - timedelta(days=early_stopping_tail_days)
        fit_rows = train.filter(pl.col("date") < val_cutoff)
        val_rows = train.filter(pl.col("date") >= val_cutoff)
        eval_set = [(val_rows.select(FEATURE_COLUMNS).to_numpy(), val_rows["label"].to_numpy())]

    X_train = fit_rows.select(FEATURE_COLUMNS).to_numpy()
    y_train = fit_rows["label"].to_numpy()
    X_test = test.select(FEATURE_COLUMNS).to_numpy()
    y_test = test["label"].to_numpy()

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=spw,
        max_delta_step=max_delta_step,
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=50 if eval_set else None,
    )
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    y_score = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)
    p_at_100 = precision_at_k(y_test, y_score, k=min(100, len(y_test)))

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "split": split_name,
                "n_estimators": model.best_iteration + 1 if eval_set else n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "scale_pos_weight": spw,
                "max_delta_step": max_delta_step,
                "early_stopping_tail_days": early_stopping_tail_days or 0,
            }
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


def load_featured(data_globs: list[str], cache_path: str, rebuild_cache: bool) -> pl.DataFrame:
    cache = Path(cache_path)
    if cache.exists() and not rebuild_cache:
        print(f"[train] loading cached features from {cache_path}")
        return pl.read_parquet(cache_path)

    raw = load_quarter(data_globs)
    labeled = add_labels(raw)
    featured = add_delta_features(labeled)

    cache.parent.mkdir(parents=True, exist_ok=True)
    featured.write_parquet(cache_path)
    print(f"[train] cached features to {cache_path}")
    return featured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", required=True, nargs="+", help="One or more glob paths to quarterly CSVs"
    )
    parser.add_argument("--cutoff", required=True, help="YYYY-MM-DD temporal split cutoff")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--cache", default="artifacts/_cache/featured.parquet")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-delta-step", type=int, default=0)
    parser.add_argument(
        "--early-stopping-tail-days",
        type=int,
        default=0,
        help="Carve this many trailing train days out as an eval set for early stopping; 0 disables it",
    )
    args = parser.parse_args()

    mlflow.set_experiment("drive-failure")

    featured = load_featured(args.data, args.cache, args.rebuild_cache)

    cutoff = date.fromisoformat(args.cutoff)
    grouped_train, grouped_test = grouped_temporal_split(featured, cutoff)
    naive_train, naive_test = naive_random_split(featured)

    hp = dict(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_delta_step=args.max_delta_step,
        early_stopping_tail_days=args.early_stopping_tail_days or None,
    )
    fit_and_evaluate(naive_train, naive_test, "xgb-naive-random", "naive_random", **hp)
    model, pr_auc = fit_and_evaluate(
        grouped_train, grouped_test, "xgb-grouped-temporal", "grouped_temporal", **hp
    )

    onnx_path = f"{args.artifacts_dir}/model.onnx"
    export_onnx(model, onnx_path)
    model.save_model(f"{args.artifacts_dir}/xgb_model.json")

    config_path = f"{args.artifacts_dir}/feature_config.json"
    config = json.loads(Path(config_path).read_text())
    config["model_version"] = f"xgb-{cutoff.isoformat()}-{pr_auc:.3f}"
    config["features"] = FEATURE_COLUMNS
    Path(config_path).write_text(json.dumps(config, indent=2))

    print(f"Wrote {onnx_path}, model_version={config['model_version']}")


if __name__ == "__main__":
    main()
