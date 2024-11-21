default: install lint test

install:
    uv lock --upgrade
    uv sync --frozen

lint:
    uv run --group lint ruff check
    uv run --group lint auto-typing-final .
    uv run --group lint ruff format
    uv run --group lint mypy .

test *args:
    uv run pytest {{ args }}

publish:
    rm -rf dist
    uv build
    uv publish --token $PYPI_TOKEN
