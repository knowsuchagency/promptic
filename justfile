# Format code
format:
    uvx ruff format promptic.py tests/ examples/

# Embed examples in README
embedme:
    npx embedme README.md

# Run tests
test: format
    uv run pytest -x -v tests/

# Run tests in parallel
test-parallel: format
    uv run pytest -v -n auto tests/

# Run a specific test
test-fn fn:
    uv run pytest -x -v -n 3 tests/test_promptic.py::{{fn}}

# Run tests with coverage
test-cov: format
    uv run pytest --cov=promptic --cov-report=term-missing tests/
