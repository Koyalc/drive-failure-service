"""Phase 7 -- drift detection: score the 2022-trained model against 2023 data,
run Evidently's data drift report, and quantify the PR-AUC degradation.

Usage:
    python -m src.pipeline.drift --data "data/Q1_2023/*.csv" "data/Q2_2023/*.csv"

Reuses train.py's ingest/label/delta-feature pipeline and its feature-computation
cache, so the 2023 features are built the same way the 2022 training features were.

Each --data glob is run through load_featured separately and concatenated afterward,
rather than combined into one load_featured call like train.py does for 2022: the
combined 2023 Q1+Q2 raw CSVs (14.5GB) are large enough that add_delta_features' sort
+ window step gets OOM-killed on a 16GB machine. Processing one quarter at a time keeps
peak memory bounded to roughly what already worked for 2022. Cost: drives spanning the
Q1/Q2 boundary get a second "cold start" 7-day delta reset instead of carrying trend
across the boundary -- a small, acceptable approximation for a one-off drift check (not
used for anything that trains a model).

Requires requirements-train.txt (not part of the serving image).
"""
import argparse
from pathlib import Path

import mlflow
import polars as pl
import xgboost as xgb
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sklearn.metrics import average_precision_score, roc_auc_score

from src.pipeline.train import FEATURE_COLUMNS, load_featured, precision_at_k


def load_featured_per_quarter(data_globs: list[str], cache_dir: str, rebuild_cache: bool) -> pl.DataFrame:
    frames = []
    for i, glob_path in enumerate(data_globs):
        cache_path = f"{cache_dir}/q{i}.parquet"
        frames.append(load_featured([glob_path], cache_path, rebuild_cache))
    return pl.concat(frames, how="diagonal")


def score(model: xgb.XGBClassifier, df: pl.DataFrame) -> dict:
    X = df.select(FEATURE_COLUMNS).to_numpy()
    y = df["label"].to_numpy()
    y_score = model.predict_proba(X)[:, 1]
    return {
        "pr_auc": average_precision_score(y, y_score),
        "roc_auc": roc_auc_score(y, y_score),
        "precision_at_100": precision_at_k(y, y_score, k=min(100, len(y))),
        "n_rows": len(y),
        "n_failures": int(y.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", required=True, nargs="+", help="Glob paths to the 2023 quarterly CSVs"
    )
    parser.add_argument("--cache-dir", default="artifacts/_cache/featured_2023")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--reference-cache", default="artifacts/_cache/featured.parquet")
    parser.add_argument("--model-path", default="artifacts/xgb_model.json")
    parser.add_argument("--report-out", default="docs/drift_report.html")
    parser.add_argument(
        "--drift-sample-size",
        type=int,
        default=200_000,
        help="Rows sampled from each of reference/current for the Evidently report -- "
        "running statistical drift tests (KS-test, PSI) over the full 38M/43M-row "
        "populations is neither necessary nor memory-feasible; a large random sample "
        "gives statistically equivalent drift conclusions. Full data is still used for "
        "the PR-AUC/ROC-AUC/precision@100 scoring above.",
    )
    args = parser.parse_args()

    mlflow.set_experiment("drive-failure")

    model = xgb.XGBClassifier()
    model.load_model(args.model_path)

    reference = pl.read_parquet(args.reference_cache)
    current = load_featured_per_quarter(args.data, args.cache_dir, args.rebuild_cache)

    metrics_2023 = score(model, current)
    print(f"[drift] 2023 holdout: {metrics_2023}")

    n = args.drift_sample_size
    ref_pd = reference.select(FEATURE_COLUMNS).sample(n=min(n, len(reference)), seed=0).to_pandas()
    cur_pd = current.select(FEATURE_COLUMNS).sample(n=min(n, len(current)), seed=0).to_pandas()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_pd, current_data=cur_pd)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(args.report_out)

    drift_result = report.as_dict()["metrics"][0]["result"]
    n_drifted = drift_result["number_of_drifted_columns"]
    share_drifted = drift_result["share_of_drifted_columns"]

    with mlflow.start_run(run_name="drift-2022-to-2023"):
        mlflow.log_metrics(
            {
                "pr_auc_2023": metrics_2023["pr_auc"],
                "roc_auc_2023": metrics_2023["roc_auc"],
                "precision_at_100_2023": metrics_2023["precision_at_100"],
                "n_drifted_columns": n_drifted,
                "share_drifted_columns": share_drifted,
            }
        )
        mlflow.log_artifact(args.report_out)

    print(
        f"[drift] {n_drifted}/{len(FEATURE_COLUMNS)} features drifted ({share_drifted:.0%})"
    )
    print(f"Wrote {args.report_out}")


if __name__ == "__main__":
    main()
