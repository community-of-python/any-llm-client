default: install lint test

install:
    uv lock --upgrade
    uv sync --frozen

lint:
    uv run ruff check
    uv run ruff format
    uv run mypy .

test *args:
    uv run pytest {{ args }}

publish:
    rm -rf dist
    uv build
    uv publish --token $PYPI_TOKEN
