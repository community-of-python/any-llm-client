default: install lint test

install:
    uv lock --upgrade
    uv sync --frozen --all-groups

lint:
    uv run --group lint ruff check
    uv run --group lint auto-typing-final .
    uv run --group lint ruff format
    uv run --group lint mypy .

_test-no-http *args:
    uv run pytest --ignore tests/test_http.py {{ args }}

test *args:
    #!/bin/bash
    uv run litestar --app tests.testing_app:app run &
    APP_PID=$!
    uv run pytest {{ args }}
    TEST_RESULT=$?
    kill $APP_PID
    wait $APP_PID 2>/dev/null
    exit $TEST_RESULT

publish:
    rm -rf dist
    uv build
    uv publish --token $PYPI_TOKEN
