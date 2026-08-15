"""Phase 1.2 — lazy-load Backblaze quarterly CSVs with Polars.

Only the columns needed for the failure model are pulled to keep memory
bounded across ~150-column source files. SMART attribute selection follows
Backblaze's own driver-failure analysis: 5, 187, 188, 197, 198 carry most of
the signal; 9, 194, 241, 242 are added for context (power-on hours,
temperature, LBA written/read).
"""
import polars as pl

SMART_ATTRIBUTES = [5, 9, 187, 188, 194, 197, 198, 241, 242]

BASE_COLUMNS = ["date", "serial_number", "model", "capacity_bytes", "failure"]

SMART_COLUMNS = [f"smart_{n}_raw" for n in SMART_ATTRIBUTES] + [
    f"smart_{n}_normalized" for n in SMART_ATTRIBUTES
]


def load_quarter(glob_path: str) -> pl.DataFrame:
    """Scan a glob of daily CSVs for one quarter and return only the needed columns.

    Logs (rather than silently dropping) any expected SMART columns missing
    from this quarter's schema -- column sets change across Backblaze data years.
    """
    lazy = pl.scan_csv(glob_path, infer_schema_length=10000)
    available = set(lazy.collect_schema().names())

    wanted = BASE_COLUMNS + SMART_COLUMNS
    missing = [c for c in wanted if c not in available]
    if missing:
        print(f"[ingest] {glob_path}: missing columns this quarter: {missing}")

    present = [c for c in wanted if c in available]
    return lazy.select(present).with_columns(pl.col("date").str.to_date()).collect(streaming=True)
