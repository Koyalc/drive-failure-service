"""Phase 1.2 — lazy-load Backblaze quarterly CSVs with Polars.

Only the columns needed for the failure model are pulled to keep memory
bounded across ~150-column source files. SMART attribute selection follows
Backblaze's own driver-failure analysis: 5, 187, 188, 197, 198 carry most of
the signal; 9, 194, 241, 242 are added for context (power-on hours,
temperature, LBA written/read).
"""
import polars as pl

from src.feature_spec import BASE_COLUMNS, SMART_ATTRIBUTES, SMART_COLUMNS

__all__ = ["SMART_ATTRIBUTES", "BASE_COLUMNS", "SMART_COLUMNS", "load_quarter"]


def load_quarter(glob_path: str | list[str]) -> pl.DataFrame:
    """Scan a glob (or list of globs, one per quarter) of daily CSVs and return needed columns.

    Logs (rather than silently dropping) any expected SMART columns missing
    from this quarter's schema -- column sets change across Backblaze data years.

    Each glob is scanned separately (not as one combined `scan_csv` call): Backblaze's
    column set can differ between quarters -- e.g. 2023 Q2 added pod_id/vault_id and two
    SMART attributes partway through the year -- and a single multi-file scan requires a
    uniform schema across every matched file, so it errors on any quarter boundary where
    the schema actually changed.
    """
    paths = [glob_path] if isinstance(glob_path, str) else glob_path
    wanted = BASE_COLUMNS + SMART_COLUMNS

    frames = []
    for path in paths:
        lazy = pl.scan_csv(path, infer_schema_length=10000)
        available = set(lazy.collect_schema().names())

        missing = [c for c in wanted if c not in available]
        if missing:
            print(f"[ingest] {path}: missing columns this quarter: {missing}")

        present = [c for c in wanted if c in available]
        frames.append(lazy.select(present).with_columns(pl.col("date").str.to_date()))

    return pl.concat(frames, how="diagonal").collect(streaming=True)
