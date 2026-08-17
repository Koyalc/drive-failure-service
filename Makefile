.PHONY: install install-train dev-model serve test lint docker-build docker-run train drift

install:
	pip install -r requirements-dev.txt

install-train:
	pip install -r requirements-train.txt

dev-model:
	python -m src.pipeline.dummy_model

train:
	python -m src.pipeline.train --data "data/Q1_2022/*.csv" "data/Q2_2022/*.csv" --cutoff 2022-05-01

drift:
	python -m src.pipeline.drift --data "data/Q1_2023/*.csv" "data/Q2_2023/*.csv"

serve:
	uvicorn src.api.main:app --reload --port 8000

test:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

docker-build:
	test -f artifacts/model.onnx || python -m src.pipeline.dummy_model
	docker build -t drive-failure:local .

docker-run:
	docker run -p 8080:8080 drive-failure:local
