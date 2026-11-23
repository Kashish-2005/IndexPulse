.PHONY: install test lint run-pipeline run-app

PYTHON := python
PIP    := pip
VENV   := .venv

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PIP) install mypy pytest-cov

test:
	pytest -q --cov=eindex --cov-report=term-missing --cov-fail-under=70

lint:
	ruff check .
	ruff format --check .
	mypy src/eindex \
		--ignore-missing-imports \
		--no-strict-optional \
		--allow-untyped-defs \
		--allow-untyped-calls

run-pipeline:
	$(PYTHON) -m eindex.pipeline

run-app:
	streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
