# Building the Documentation

The chatlas documentation site is built with [Quarto](https://quarto.org/) and
[quartodoc](https://machow.github.io/quartodoc/), and published to GitHub Pages.

## Prerequisites

1. **Quarto** must be installed on your system. See the
   [Quarto install guide](https://quarto.org/docs/get-started/). CI currently
   pins version `1.9.37`.
2. **The `docs` dependency extra** from `pyproject.toml` must be installed.
   `quartodoc` and the other doc-building tools live in the `docs`
   optional-dependency group (see `[project.optional-dependencies]`
   in `pyproject.toml`), *not* the base dependencies. The simplest way to
   get everything is:

   ```bash
   make setup   # runs `uv sync --all-extras`
   ```

   Note, the `make quartodoc` target (and `make docs` / `make docs-preview`, which
   depend on it) list `setup` as a prerequisite, so they run `uv sync
   --all-extras` for you before building — you don't have to run `make setup`
   by hand. You still need Quarto itself installed separately.

## How it fits together

- **quartodoc** reads docstrings from the `chatlas/` package and generates the
  API reference pages under `docs/reference/`, plus the `docs/_sidebar.yml`
  navigation file. Its configuration is the `quartodoc:` section of
  `docs/_quarto.yml`, which lists every symbol to document.
- **Quarto** renders the whole site (guides in `docs/*.qmd`, plus the generated
  reference) into a static website. Site config lives in `docs/_quarto.yml`.
- **interlinks** cross-link to Python, Pydantic, and chatlas's own API objects.

## Common commands

```bash
make quartodoc      # generate API reference stubs + interlinks (prereq for the rest)
make docs           # quartodoc, then `quarto render docs`  → build the full site
make docs-preview   # quartodoc, then `quarto preview docs` → live-serve locally
```

`make docs` and `make docs-preview` both depend on `make quartodoc`, so you
normally just run one of them. All three also depend on `setup`, so they run
`uv sync --all-extras` to install/refresh dependencies before building — you
don't need to run `make setup` first.

## Executable code cells

Several `.qmd` guides contain executable Python cells that make real API calls
(OpenAI, Anthropic). Locally you'll need the relevant API keys set in your
environment to render those pages from scratch. In CI these outputs are cached
via Quarto's [freeze](https://quarto.org/docs/projects/code-execution.html#freeze)
mechanism (`docs/_freeze/`).

## Publishing (CI)

Two GitHub Actions workflows handle the deployed site:

- **`.github/workflows/docs-publish.yml`** — on push to `main` (when `docs/**`
  changes) or manual dispatch. Installs Quarto + the `docs` extra, runs
  `make quartodoc`, then renders and publishes to the `gh-pages` branch
  (served at <https://posit-dev.github.io/chatlas/>).
- **`.github/workflows/docs-freeze.yml`** — manual dispatch only. Re-renders the
  docs to refresh the freeze cache (`docs/_freeze/`) and commits it back, so the
  publish job doesn't need to re-execute code cells that hit live APIs.
