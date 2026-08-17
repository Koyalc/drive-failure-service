# Drive Failure Prediction Service

Predicts hard-drive failure within a 30-day window from SMART telemetry, trained on
Backblaze's public hard-drive dataset. Served via FastAPI + ONNX Runtime, containerized,
deployed to Cloud Run.

**Status: Phases 1-3 complete.** Trained on real 2022 Q1-Q2 Backblaze data (2023 Q1-Q2 held out
for the Phase 7 drift measurement), iterated past the leakage baseline with real feature
engineering, and containerized at 414MB. Cloud deploy, monitoring, and drift detection are
still ahead — see the milestone list in the build guide.

## Quick start

```bash
make install          # dev deps (serving + training + test)
make dev-model         # generates a placeholder artifacts/model.onnx for local dev
make serve              # http://localhost:8000
make test
```

```bash
# smart_N_raw_7d_ago is that attribute's raw reading from ~7 days earlier -- the
# caller supplies it so the server can derive the same trend features training used.
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "records": [{
    "serial_number": "Z1TEST0001", "capacity_bytes": 4000787030016,
    "smart_5_raw": 0, "smart_5_normalized": 100, "smart_5_raw_7d_ago": 0,
    "smart_9_raw": 12000, "smart_9_normalized": 90, "smart_9_raw_7d_ago": 11832,
    "smart_187_raw": 0, "smart_187_normalized": 100, "smart_187_raw_7d_ago": 0,
    "smart_188_raw": 0, "smart_188_normalized": 100, "smart_188_raw_7d_ago": 0,
    "smart_194_raw": 30, "smart_194_normalized": 65, "smart_194_raw_7d_ago": 29,
    "smart_197_raw": 0, "smart_197_normalized": 100, "smart_197_raw_7d_ago": 0,
    "smart_198_raw": 0, "smart_198_normalized": 100, "smart_198_raw_7d_ago": 0,
    "smart_241_raw": 500000, "smart_241_normalized": 100, "smart_241_raw_7d_ago": 498000,
    "smart_242_raw": 500000, "smart_242_normalized": 100, "smart_242_raw_7d_ago": 498000
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
  feature_spec.py  SMART attribute list + feature naming, shared by training and serving
                    (dependency-free -- keeps polars/xgboost out of the serving image)
  api/             FastAPI app, Pydantic schemas, ONNX predictor wrapper, Prometheus metrics
  pipeline/        ingest (Polars), labeling + trend features + leakage-safe split, training
tests/             API tests (dummy-model fixture), predictor unit test, XGBoost/ONNX parity check
artifacts/         model.onnx (gitignored) + feature_config.json (tracked; feature list is
                    written by train.py so serving can never drift out of sync with training)
```

## Results

Trained on 2022 Q1-Q2 (38M drive-days, 90 days x2 quarters, ~1,300 failures — a 0.003% positive
rate). Cutoff for the temporal split: 2022-05-01.

**Why not "accuracy"?** A model predicting "never fails" for every drive scores 99.997%
accuracy on this data while catching zero real failures — accuracy is meaningless under this
level of class imbalance. PR-AUC and precision@k are reported instead because they reflect
what an operator actually cares about: of the drives flagged as high-risk, how many really fail.

| Metric | Naive random split | Grouped temporal split |
|---|---|---|
| PR-AUC | 0.279 | **0.101** |
| ROC-AUC | 0.939 | 0.635 |
| Precision@100 | 95% | 22% |

The naive split lets the model see the future and memorize per-drive SMART baselines it
implicitly re-encounters in "test" rows from the same serial number. Once both leaks are
closed, PR-AUC drops to a third of the naive number, and precision@100 — "of the 100 drives
we'd flag tonight, how many actually fail" — drops from 95 to 22. The naive number is not a
result anyone should ship; it's the control that shows why the split matters. See
`src/pipeline/features.py` for both split implementations and `src/pipeline/train.py --help`
for the run that produced this table. A visual version of this ablation and the iteration
table below is in [`docs/ablation_comparison.html`](docs/ablation_comparison.html).

**Iterating on the grouped-temporal (honest) split, in order:**

| Change | PR-AUC | Precision@100 | Verdict |
|---|---|---|---|
| Raw SMART snapshot only, depth 6, 400 trees | 0.027 | 6% | Phase 1 baseline |
| + 7-day trend features, depth 7, early-stopped on a 14-day tail | 0.010 | 3% | **Worse** — deeper trees (575) overfit a validation window with too few failures in it to be a reliable stopping signal |
| + 7-day trend features, same depth/trees as baseline (isolates the feature effect) | 0.038 | 6% | Trend features alone help (+40% PR-AUC) |
| + early-stopped on a 30-day tail (more stable), depth 6 | **0.101** | **22%** | Best — stopped at 130 trees once the larger validation window showed genuine overfitting |
| + lower learning rate (0.03), more rounds | 0.068 | 26% | Worse PR-AUC despite a marginally better precision@100; not kept |

The failed middle attempt is left in the table on purpose: adding both model capacity and new
features in the same step made it impossible to tell whether the trend features helped or the
extra depth was quietly overfitting. Separating them showed the features were good and the
capacity increase alone was the problem. Reproduce with:
`python -m src.pipeline.train --data "data/Q1_2022/*.csv" "data/Q2_2022/*.csv" --cutoff 2022-05-01 --max-depth 6 --n-estimators 600 --early-stopping-tail-days 30`

| | |
|---|---|
| Image size | 414MB |
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
  memorize a specific drive's SMART baseline. See the ablation above.
- **`sympy`/`mpmath` stripped from the image** — `onnxruntime` declares them as a hard
  dependency for optional symbolic shape inference that plain `InferenceSession.run()` never
  exercises. Removing them (in the Dockerfile's builder stage, *before* the final-stage COPY —
  deleting after copying only masks the layer, it doesn't shrink the image) cut ~85MB, taking
  the image from 524MB to 414MB. Verified the container still serves correctly afterward.

## Reproduce

```bash
make install-train                                    # training deps
python -m src.pipeline.train --data "data/Q1_2022/*.csv" "data/Q2_2022/*.csv" --cutoff 2022-05-01
make docker-build
make docker-run
```

## Deploy to Cloud Run

CI builds and pushes the image to GHCR on every push to `master`; the `deploy` job then ships
that same image straight to Cloud Run (no separate Artifact Registry push — GHCR is free for
public images and Cloud Run can pull from any registry the image is publicly readable from).
`--min-instances=0 --allow-unauthenticated` keeps it inside the Cloud Run Always Free tier and
reachable for a demo curl; cold starts are the accepted tradeoff (see Design decisions).

One-time setup (outside this repo, in your GCP project):

```bash
gcloud services enable run.googleapis.com
gcloud iam service-accounts create drive-failure-deployer
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:drive-failure-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:drive-failure-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
gcloud iam service-accounts keys create key.json \
  --iam-account=drive-failure-deployer@$PROJECT_ID.iam.gserviceaccount.com
```

Then, in the GitHub repo: Settings → Secrets and variables → Actions, add `GCP_PROJECT_ID`,
`GCP_REGION` (e.g. `us-central1`), and `GCP_SA_KEY` (the contents of `key.json` — delete the
local copy after). Also flip the GHCR package's visibility to public once
(Package settings on GitHub — Cloud Run needs to pull it anonymously). Delete `key.json` locally
after pasting it into the secret; it's a long-lived credential and shouldn't sit on disk.
