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

.PHONY: check-tests-vcr
check-tests-vcr:  ## [py] Run VCR-compatible tests (skips VCR-incompatible providers/tests)
	@echo ""
	@echo "📼 Running VCR-compatible tests"
	TEST_BEDROCK=false \
	TEST_SNOWFLAKE=false \
	TEST_GOOGLE_STREAMING=false \
	TEST_PORTKEY=false \
	TEST_GITHUB=false \
	TEST_HUGGINGFACE=false \
	TEST_MISTRAL=false \
	TEST_OPENROUTER=false \
	uv run pytest tests/ -v \
		--ignore=tests/test_batch_chat.py \
		--ignore=tests/test_inspect.py \
		--ignore=tests/test_mcp_client.py

.PHONY: check-tests-live
check-tests-live:  ## [py] Run all tests with live APIs (requires credentials)
	@echo ""
	@echo "🔴 Running live API tests"
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
record-vcr: record-vcr-providers record-vcr-chat ## [py] Record all VCR cassettes

.PHONY: record-vcr-providers
record-vcr-providers: record-vcr-openai record-vcr-anthropic record-vcr-google record-vcr-azure record-vcr-cloudflare record-vcr-databricks record-vcr-deepseek record-vcr-github record-vcr-huggingface record-vcr-mistral record-vcr-openai-completions record-vcr-openrouter record-vcr-portkey ## [py] Record VCR cassettes for providers

.PHONY: record-vcr-openai
record-vcr-openai:
	@echo "📼 Recording OpenAI cassettes"
	uv run pytest --record-mode=all tests/test_provider_openai.py -v

.PHONY: record-vcr-anthropic
record-vcr-anthropic:
	@echo "📼 Recording Anthropic cassettes"
	uv run pytest --record-mode=all tests/test_provider_anthropic.py -v

.PHONY: record-vcr-google
record-vcr-google:
	@echo "📼 Recording Google cassettes"
	uv run pytest --record-mode=all tests/test_provider_google.py -v

.PHONY: record-vcr-azure
record-vcr-azure:
	@echo "📼 Recording Azure cassettes"
	uv run pytest --record-mode=all tests/test_provider_azure.py -v

.PHONY: record-vcr-cloudflare
record-vcr-cloudflare:
	@echo "📼 Recording Cloudflare cassettes"
	uv run pytest --record-mode=all tests/test_provider_cloudflare.py -v

.PHONY: record-vcr-databricks
record-vcr-databricks:
	@echo "📼 Recording Databricks cassettes"
	uv run pytest --record-mode=all tests/test_provider_databricks.py -v

.PHONY: record-vcr-deepseek
record-vcr-deepseek:
	@echo "📼 Recording DeepSeek cassettes"
	uv run pytest --record-mode=all tests/test_provider_deepseek.py -v

.PHONY: record-vcr-github
record-vcr-github:
	@echo "📼 Recording GitHub cassettes"
	uv run pytest --record-mode=all tests/test_provider_github.py -v

.PHONY: record-vcr-huggingface
record-vcr-huggingface:
	@echo "📼 Recording HuggingFace cassettes"
	uv run pytest --record-mode=all tests/test_provider_huggingface.py -v

.PHONY: record-vcr-mistral
record-vcr-mistral:
	@echo "📼 Recording Mistral cassettes"
	uv run pytest --record-mode=all tests/test_provider_mistral.py -v

.PHONY: record-vcr-openai-completions
record-vcr-openai-completions:
	@echo "📼 Recording OpenAI Completions cassettes"
	uv run pytest --record-mode=all tests/test_provider_openai_completions.py -v

.PHONY: record-vcr-openrouter
record-vcr-openrouter:
	@echo "📼 Recording OpenRouter cassettes"
	uv run pytest --record-mode=all tests/test_provider_openrouter.py -v

.PHONY: record-vcr-portkey
record-vcr-portkey:
	@echo "📼 Recording Portkey cassettes"
	uv run pytest --record-mode=all tests/test_provider_portkey.py -v

.PHONY: record-vcr-chat
record-vcr-chat:
	@echo "📼 Recording chat cassettes"
	uv run pytest --record-mode=all \
		tests/test_chat.py \
		tests/test_chat_dangling_tools.py \
		tests/test_parallel_chat.py \
		tests/test_parallel_chat_improved.py \
		tests/test_parallel_chat_errors.py \
		tests/test_parallel_chat_ordering.py \
		tests/test_tokens.py \
		-v

.PHONY: rerecord-vcr
rerecord-vcr:  ## [py] Delete and re-record all VCR cassettes
	@echo "🗑️  Deleting existing cassettes"
	rm -rf tests/_vcr/test_provider_* tests/_vcr/test_chat* tests/_vcr/test_parallel_chat*
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
