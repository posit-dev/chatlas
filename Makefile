.PHONY: docs
docs:  ## [docs] Build the documentation
	quarto render docs

.PHONY: docs-preview
docs-preview:  ## [docs] Preview the documentation
	quarto preview docs

.PHONY: py-setup
py-setup:  ## [py] Setup python environment
	uv sync --all-extras

.PHONY: py-check
py-check:  py-check-tests py-check-format py-check-types ## [py] Run python checks

.PHONY: py-check-tox
py-check-tox:  ## [py] Run python 3.9 - 3.12 checks with tox
	@echo ""
	@echo "🔄 Running tests and type checking with tox for Python 3.9--3.12"
	uv run tox run-parallel

.PHONY: py-check-tests
py-check-tests:  ## [py] Run python tests
	@echo ""
	@echo "🧪 Running tests with pytest"
	uv run pytest

.PHONY: py-check-types
py-check-types:  ## [py] Run python type checks
	@echo ""
	@echo "📝 Checking types with pyright"
	uv run pyright

.PHONY: py-check-format
py-check-format:
	@echo ""
	@echo "📐 Checking format with ruff"
	uv run ruff check chatlas --config pyproject.toml

.PHONY: py-format
py-format: ## [py] Format python code
	uv run ruff check --fix chatlas --config pyproject.toml
	uv run ruff format chatlas --config pyproject.toml

.PHONY: py-update-snaps
py-update-snaps:  ## [py] Update python test snapshots
	@echo "📸 Updating pytest snapshots"
	uv run pytest --snapshot-update

.PHONY: help
help:  ## Show help messages for make targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; { \
		printf "\033[32m%-18s\033[0m", $$1; \
		if ($$2 ~ /^\[docs\]/) { \
			printf "\033[34m[docs]\033[0m%s\n", substr($$2, 7); \
		} else if ($$2 ~ /^\[py\]/) { \
			printf "  \033[33m[py]\033[0m%s\n", substr($$2, 5); \
		} else if ($$2 ~ /^\[r\]/) { \
			printf "   \033[31m[r]\033[0m%s\n", substr($$2, 4); \
		} else { \
			printf "       %s\n", $$2; \
		} \
	}'

.DEFAULT_GOAL := help
