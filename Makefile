.PHONY: setup download-data train serve dashboard test lint format docker-up clean

setup:
	pip install -e ".[dev]"
	pre-commit install

download-data:
	python -m src.data.download

train:
	python scripts/train.py

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/

docker-up:
	docker-compose up --build

clean:
	rm -rf data/raw/* data/processed/* mlruns/ .pytest_cache/ __pycache__/
