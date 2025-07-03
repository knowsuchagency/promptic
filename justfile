# Format code
format:
    uvx ruff format promptic.py tests/ examples/

# Embed examples in README
embedme:
    npx embedme README.md

# Run tests
test skip_examples="true": format
    #!/usr/bin/env bash
    if [ "{{skip_examples}}" = "true" ]; then
        uv run pytest -x -v --record-mode=once -m "not examples" tests/
    else
        uv run pytest -x -v --record-mode=once tests/
    fi

# Run a specific test
test-fn fn:
    uv run pytest -x -v -n 3 tests/test_promptic.py::{{fn}}

# Run tests with coverage
test-cov: format
    uv run pytest --cov=promptic --cov-report=term-missing tests/

pre-commit:
    uv run pre-commit

publish:
    rm -rf dist/* ; uv build ; uv publish

# Run tests with VCR recording
test-record: format
    uv run pytest -x -v --record-mode=rewrite tests/

# Run tests with VCR in replay-only mode (for CI)
test-ci: format
    uv run pytest -x -v --record-mode=none tests/

# Update specific VCR cassettes
update-cassette pattern:
    uv run pytest -x -v --record-mode=rewrite -k "{{pattern}}" tests/
