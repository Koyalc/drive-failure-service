# Drive Failure Prediction Service

Predicts hard-drive failure within a 30-day window from SMART telemetry, trained on
Backblaze's public hard-drive dataset. Served via FastAPI + ONNX Runtime, containerized,
deployed to Cloud Run.

**Status: scaffolding stage.** The API, tests, Docker build, and CI are wired up end-to-end
against a placeholder (randomly-weighted) ONNX model so the whole pipeline is runnable before
real training data is downloaded. Sections below get filled in as each phase completes —
see `docs/` for the leakage ablation and drift report once they exist.

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

_To fill in after Phase 1 training runs on real data:_

| Metric | Naive random split | Grouped temporal split |
|---|---|---|
| PR-AUC | — | — |
| Precision@100 | — | — |

| | |
|---|---|
| Image size | — |
| p50 / p95 / p99 latency @ 200 RPS | — |
| Drift: 2022→2023 PR-AUC degradation | — |

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
python -m src.pipeline.train --data "data/2022*/*.csv" --cutoff 2022-05-01
make docker-build
make docker-run
```
