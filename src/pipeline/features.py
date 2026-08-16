"""Phase 1.3/1.4 — labeling and the leakage-safe train/test split.

Label: will this drive fail within the next 30 days?
Split: time-based cut, then drop any serial_number that spans the boundary
so no drive leaks between train and test. `naive_random_split` exists only
to produce the comparison ablation described in the README -- never use it
for a model you intend to trust.
"""
from datetime import date

import polars as pl

from src.feature_spec import SMART_ATTRIBUTES, delta7d_column


def add_delta_features(df: pl.DataFrame) -> pl.DataFrame:
    """Trailing 7-day change in each raw SMART counter, per drive.

    A single day's snapshot can't distinguish a reading that's been flat for
    months from one climbing fast -- most SMART-based failure models treat
    the growth rate of reallocated/pending-sector counts as more informative
    than their absolute value. Requires the frame to already contain a full,
    date-contiguous history per serial_number (not a single day).
    """
    df = df.sort(["serial_number", "date"])
    exprs = [
        (pl.col(f"smart_{n}_raw") - pl.col(f"smart_{n}_raw").shift(7).over("serial_number"))
        .fill_null(0.0)
        .alias(delta7d_column(n))
        for n in SMART_ATTRIBUTES
    ]
    return df.with_columns(exprs)


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
