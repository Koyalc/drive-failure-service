# Drive Failure Prediction Service

Predicts hard-drive failure within a 30-day window from SMART telemetry, trained on
Backblaze's public hard-drive dataset. Served via FastAPI + ONNX Runtime, containerized,
deployed to Cloud Run.

**Status: Phase 1 complete.** Trained on real 2022 Q1-Q2 Backblaze data (2023 Q1-Q2 held out
for the Phase 7 drift measurement); the API is serving that model's real ONNX export, not a
placeholder. Cloud deploy, monitoring, and drift detection are still ahead — see the milestone
list in the build guide.

## Quick start

```bash
make install          # dev deps (serving + training + test)
make dev-model         # generates a placeholder artifacts/model.onnx for local dev
make serve              # http://localhost:8000
make test
```

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "records": [{
    "serial_number": "Z1TEST0001", "capacity_bytes": 4000787030016,
    "smart_5_raw": 0, "smart_5_normalized": 100,
    "smart_9_raw": 12000, "smart_9_normalized": 90,
    "smart_187_raw": 0, "smart_187_normalized": 100,
    "smart_188_raw": 0, "smart_188_normalized": 100,
    "smart_194_raw": 30, "smart_194_normalized": 65,
    "smart_197_raw": 0, "smart_197_normalized": 100,
    "smart_198_raw": 0, "smart_198_normalized": 100,
    "smart_241_raw": 500000, "smart_241_normalized": 100,
    "smart_242_raw": 500000, "smart_242_normalized": 100
  }]
}'
```

## Architecture

```
Backblaze CSVs → Polars pipeline → XGBoost → ONNX → FastAPI → Docker → Cloud Run
                                      ↓                            ↓
                                   MLflow                    Prometheus + Evidently
```

## SMART attribute selection

Backblaze's own failure analysis found five attributes carry most of the signal:
5 (reallocated sectors), 187 (reported uncorrectable errors), 188 (command timeout),
197 (current pending sector), 198 (offline uncorrectable). This service adds 9
(power-on hours), 194 (temperature), and 241/242 (LBA written/read) for context.
See `src/pipeline/ingest.py`.

## Repo layout

```
src/
  api/           FastAPI app, Pydantic schemas, ONNX predictor wrapper, Prometheus metrics
  pipeline/       ingest (Polars), labeling + leakage-safe split, XGBoost training, ONNX export
tests/            API tests (dummy-model fixture), predictor unit test, XGBoost/ONNX parity check
artifacts/        model.onnx (gitignored) + feature_config.json (tracked)
```

## Results

Trained on 2022 Q1-Q2 (38M drive-days, 90 days x2 quarters, ~1,300 failures — a 0.003% positive
rate). Cutoff for the temporal split: 2022-05-01.

| Metric | Naive random split | Grouped temporal split |
|---|---|---|
| PR-AUC | 0.222 | **0.027** |
| ROC-AUC | 0.947 | 0.690 |
| Precision@100 | 86% | 6% |

The naive split lets the model see the future and memorize per-drive SMART baselines it
implicitly re-encounters in "test" rows from the same serial number. Once both leaks are
closed, PR-AUC drops **8x** and precision@100 — "of the 100 drives we'd flag tonight, how many
actually fail" — drops from 86 to 6. The naive number is not a result anyone should ship;
it's the control that shows why the split matters. See `src/pipeline/features.py` for both
split implementations and `src/pipeline/train.py` for the run that produced this table.

The 0.027 PR-AUC on the grouped split is a real baseline, not a bug — expect it to improve with
feature engineering (rolling SMART deltas, drive age) in a later iteration; it is not this
project's focus and is being reported honestly rather than hidden.

| | |
|---|---|
| Image size | — (Phase 3) |
| p50 / p95 / p99 latency @ 200 RPS | — (Phase 8) |
| Drift: 2022→2023 PR-AUC degradation | — (Phase 7) |

## Design decisions

- **ONNX over serving XGBoost/PyTorch directly** — `requirements.txt` (the serving image)
  pulls in only `onnxruntime`, not `xgboost`/`polars`/`mlflow`, keeping the container small.
  Training deps live in `requirements-train.txt`.
- **`min-instances=0` on Cloud Run** — stays inside the Always Free tier; accepted cold-start
  latency as the tradeoff.
- **PR-AUC and precision@k, not accuracy** — failures are well under 1% of rows; a model that
  always predicts "no failure" scores ~99.9% accuracy while being useless operationally.
- **Grouped temporal split, not random** — a random split lets the model see the future and
  memorize a specific drive's SMART baseline. See the ablation above once run.

## Reproduce

```bash
make install-train                                    # training deps
python -m src.pipeline.train --data "data/Q1_2022/*.csv" "data/Q2_2022/*.csv" --cutoff 2022-05-01
make docker-build
make docker-run
```
