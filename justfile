format:
    uvx ruff format promptic.py tests/ examples/

embedme:
    npx embedme README.md

test: format
    uv run pytest -x -v tests/

test-parallel: format
    uv run pytest -v -n auto tests/

test-fn fn:
    uv run pytest -v -n auto tests/test_promptic.py::{{fn}}

test-cov: format
    uv run pytest --cov=promptic --cov-report=term-missing tests/
