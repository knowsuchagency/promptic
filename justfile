format:
    uvx ruff format promptic.py tests/

test: format
    uv run pytest -v tests/
