# VCR Cassettes Directory

This directory contains VCR.py cassette files that record HTTP interactions during test runs.

## What are cassettes?

Cassettes are YAML files that store recorded HTTP request/response pairs. They allow tests to:
- Run without making actual API calls
- Execute much faster than live API tests
- Work offline or in environments without API access
- Avoid API rate limits and costs

## Recording new cassettes

To record new cassettes or update existing ones:

```bash
# Record all tests (overwrites existing cassettes)
uv run pytest --record-mode=rewrite

# Record only new interactions (default)
uv run pytest --record-mode=once

# Record specific test
uv run pytest tests/test_promptic.py::test_basic --record-mode=rewrite
```

## Important notes

1. **API Keys**: Sensitive headers are automatically filtered by the vcr_config in conftest.py
2. **Determinism**: Tests should use temperature=0 for consistent LLM responses
3. **Version Control**: Cassettes should be committed to git for CI/CD
4. **Updates**: Re-record cassettes when prompts or expected outputs change significantly

## Cassette organization

Cassettes are organized by test module name:
- `test_promptic/` - Main test suite cassettes
- `test_examples/` - Example test cassettes (if any)

Each cassette filename matches the test function name.
