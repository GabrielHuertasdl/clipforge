.DEFAULT_GOAL := help
UV := uv

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install dependencies (needs uv + ffmpeg)
	@command -v ffmpeg >/dev/null || (echo "ffmpeg not found on PATH — see README"; exit 1)
	$(UV) venv --python 3.11
	$(UV) pip install -e ".[dev]"
	@mkdir -p input/raw input/manual output/pending output/approved output/rejected output/thumbs data/cache data/logs

test: ## Run the test suite (installs dev extras first, so it works from a clean checkout)
	$(UV) sync --all-extras --dev
	$(UV) run pytest -q

lint: ## Ruff check + format check
	$(UV) run ruff check clipforge tests
	$(UV) run ruff format --check clipforge tests

fmt: ## Ruff autofix + format
	$(UV) run ruff check --fix clipforge tests
	$(UV) run ruff format clipforge tests

run: ## Scan input/raw/ for new videos
	$(UV) run clipforge scan

clean: ## Remove caches and temp files
	rm -rf data/cache/tmp/* .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: help setup test lint fmt run clean
