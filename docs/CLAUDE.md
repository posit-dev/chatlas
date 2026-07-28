# Documentation

Documentation is built with Quarto and quartodoc:
- API reference generated from docstrings in `chatlas/` modules
- Guides and examples in `docs/` as `.qmd` files
- Type definitions in `chatlas/types/` provide provider-specific parameter types

Build commands: `make docs` (build), `make docs-preview` (serve locally), or
`make quartodoc` (regenerate the API reference stubs only). All three require
[Quarto](https://quarto.org/docs/get-started/) installed on your system.
On a Mac, Quarto can be installed with homebrew (`quarto` cask).

Note: `quartodoc` lives in the `docs` optional-dependency extra (not the base
deps), so it isn't present after a plain `uv sync`. The doc targets depend on
`setup` (`uv sync --all-extras`), so they install the extra automatically before
running. See `docs/dev/building-docs.md` for the full build/publish walkthrough.
