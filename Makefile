.PHONY: setup
setup:  ## [py] Setup python environment
	uv sync --all-extras

.PHONY: build
build:   ## [py] Build python package
	@echo "🧳 Building python package"
	@[ -d dist ] && rm -r dist || true
	uv build

.PHONY: check
check: check-format check-types check-tests ## [py] Run python checks

.PHONY: check-tests
check-tests:  ## [py] Run python tests
	@echo ""
	@echo "🧪 Running tests with pytest"
	uv run pytest

.PHONY: check-types
check-types:  ## [py] Run python type checks
	@echo ""
	@echo "📝 Checking types with pyright"
	uv run pyright

.PHONY: check-format
check-format:
	@echo ""
	@echo "📐 Checking format with ruff"
	uv run ruff check chatlas --config pyproject.toml

.PHONY: format
format: ## [py] Format python code
	uv run ruff check --fix chatlas --config pyproject.toml
	uv run ruff format chatlas --config pyproject.toml

.PHONY: check-tox
check-tox:  ## [py] Run python 3.9 - 3.12 checks with tox
	@echo ""
	@echo "🔄 Running tests and type checking with tox for Python 3.9--3.12"
	uv run tox run-parallel

.PHONY: docs
docs: quartodoc
	quarto render docs

.PHONY: docs-preview
docs-preview: quartodoc
	quarto preview docs

.PHONY: quartodoc
quartodoc: 
	@echo "📖 Generating python docs with quartodoc"
	@$(eval export IN_QUARTODOC=true)
	cd docs && uv run quartodoc build
	cd docs && uv run quartodoc interlinks

.PHONY: quartodoc-watch
quartodoc-watch:
	@echo "📖 Generating python docs with quartodoc"
	@$(eval export IN_QUARTODOC=true)
	uv run quartodoc build --config docs/_quarto.yml --watch

.PHONY: update-snaps
update-snaps:
	@echo "📸 Updating pytest snapshots"
	uv run pytest --snapshot-update

.PHONY: record-vcr
record-vcr: record-vcr-openai record-vcr-anthropic record-vcr-google ## [py] Record VCR cassettes for all providers

.PHONY: record-vcr-openai
record-vcr-openai:  ## [py] Record VCR cassettes for OpenAI
	@echo "📼 Recording OpenAI cassettes"
	uv run pytest --record-mode=all tests/test_provider_openai.py -v

.PHONY: record-vcr-anthropic
record-vcr-anthropic:  ## [py] Record VCR cassettes for Anthropic
	@echo "📼 Recording Anthropic cassettes"
	uv run pytest --record-mode=all tests/test_provider_anthropic.py -v

.PHONY: record-vcr-google
record-vcr-google:  ## [py] Record VCR cassettes for Google
	@echo "📼 Recording Google cassettes"
	uv run pytest --record-mode=all tests/test_provider_google.py -v

.PHONY: rerecord-vcr
rerecord-vcr:  ## [py] Delete and re-record all VCR cassettes
	@echo "🗑️  Deleting existing cassettes"
	rm -rf tests/_vcr/test_provider_openai tests/_vcr/test_provider_anthropic tests/_vcr/test_provider_google
	$(MAKE) record-vcr

.PHONY: update-types
update-types:
	@echo "📝 Updating chat provider types"
	uv run python scripts/main.py

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
