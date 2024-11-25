default: install lint test

install:
    uv lock --upgrade
    uv sync --frozen --all-groups

lint:
    uv run --group lint ruff check
    uv run --group lint auto-typing-final .
    uv run --group lint ruff format
    uv run --group lint mypy .

test *args:
    uv run pytest {{ args }}

test-http *args:
    #!/bin/bash
    uv run litestar --app tests.http.testing_app:app run &
    APP_PID=$!
    uv run pytest tests/http/integration.py --no-cov {{ args }}
    TEST_RESULT=$?
    kill $APP_PID
    wait $APP_PID 2> /dev/null
    exit $TEST_RESULT

publish:
    rm -rf dist
    uv build
    uv publish --token $PYPI_TOKEN
