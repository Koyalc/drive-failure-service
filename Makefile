.PHONY: install install-train dev-model serve test lint docker-build docker-run

install:
	pip install -r requirements-dev.txt

install-train:
	pip install -r requirements-train.txt

dev-model:
	python -m src.pipeline.dummy_model

train:
	python -m src.pipeline.train --data "data/2022*/*.csv" --cutoff 2022-05-01

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
