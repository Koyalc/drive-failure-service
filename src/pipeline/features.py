"""Phase 1.3/1.4 — labeling and the leakage-safe train/test split.

Label: will this drive fail within the next 30 days?
Split: time-based cut, then drop any serial_number that spans the boundary
so no drive leaks between train and test. `naive_random_split` exists only
to produce the comparison ablation described in the README -- never use it
for a model you intend to trust.
"""
from datetime import date

import polars as pl


def add_labels(df: pl.DataFrame, horizon_days: int = 30) -> pl.DataFrame:
    failures = df.filter(pl.col("failure") == 1).select(
        ["serial_number", pl.col("date").alias("fail_date")]
    )
    return (
        df.join(failures, on="serial_number", how="left")
        .with_columns(
            label=(
                (pl.col("fail_date") - pl.col("date")).dt.total_days().is_between(0, horizon_days)
            )
            .fill_null(False)
            .cast(pl.Int8)
        )
        .drop("fail_date")
    )


def grouped_temporal_split(
    labeled: pl.DataFrame, cutoff: date
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = labeled.filter(pl.col("date") < cutoff)
    test = labeled.filter(pl.col("date") >= cutoff)

    overlap = set(train["serial_number"]) & set(test["serial_number"])
    if overlap:
        test = test.filter(~pl.col("serial_number").is_in(list(overlap)))
    return train, test


def naive_random_split(
    labeled: pl.DataFrame, test_fraction: float = 0.2, seed: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Random row split -- ignores both drive and temporal leakage. For the ablation only."""
    shuffled = labeled.sample(fraction=1.0, shuffle=True, seed=seed)
    cut = int(len(shuffled) * (1 - test_fraction))
    return shuffled[:cut], shuffled[cut:]
