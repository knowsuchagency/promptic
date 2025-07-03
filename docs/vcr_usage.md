# Using VCR.py with Promptic

VCR.py integration allows you to record and replay HTTP interactions in your tests, making them faster, more reliable, and cost-effective.

## Overview

When you run tests with VCR enabled, the first run records all HTTP interactions (API calls to OpenAI, Anthropic, etc.) to YAML files called "cassettes". Subsequent test runs replay these recordings instead of making real API calls.

## Benefits

- **Speed**: Tests run 10-100x faster using recordings
- **Cost**: No API charges after initial recording
- **Reliability**: Tests work offline and aren't affected by API outages
- **Determinism**: Same responses every time, making tests predictable

## Basic Usage

### 1. Mark tests with `@pytest.mark.vcr`

```python
import pytest
from promptic import llm

@pytest.mark.vcr
def test_llm_call():
    @llm(model="gpt-4o-mini", temperature=0)
    def get_greeting(name: str) -> str:
        """Say hello to {name}"""

    result = get_greeting("Alice")
    assert "Alice" in result
```

### 2. Run tests to record cassettes

```bash
# First run - records API calls
uv run pytest tests/test_vcr_example.py

# Subsequent runs - uses recordings
uv run pytest tests/test_vcr_example.py
```

## Recording Modes

Control how VCR handles recordings with the `--record-mode` option:

```bash
# Record new interactions only (default)
uv run pytest --record-mode=once

# Re-record all interactions
uv run pytest --record-mode=rewrite

# Never record, only use existing cassettes
uv run pytest --record-mode=none

# Always record, never replay
uv run pytest --record-mode=all
```

## Best Practices

### 1. Use deterministic settings

Always set `temperature=0` for consistent responses:

```python
@llm(model="gpt-4o-mini", temperature=0)  # Deterministic
def my_function():
    """..."""
```

### 2. Organize tests by recording needs

```python
# Tests that should be recorded
@pytest.mark.vcr
def test_api_interaction():
    # Makes real API calls on first run
    pass

# Tests that shouldn't be recorded
@pytest.mark.no_vcr
def test_pure_logic():
    # No API calls here
    pass
```

### 3. Update cassettes when needed

When you change prompts or expected outputs:

```bash
# Re-record specific test
uv run pytest tests/test_module.py::test_function --record-mode=rewrite

# Re-record all tests in a module
uv run pytest tests/test_module.py --record-mode=rewrite
```

### 4. Handle sensitive data

API keys are automatically filtered by the configuration in `tests/conftest.py`. The cassettes will show `REDACTED` instead of actual keys.

### 5. Commit cassettes to version control

Cassettes should be committed so that:
- CI/CD can run tests without API keys
- Other developers can run tests immediately
- You have a history of API response changes

## Advanced Usage

### Custom VCR configuration per test

```python
@pytest.mark.vcr(
    match_on=["method", "uri"],  # Don't match on body
    record_mode="new_episodes",   # Record new requests only
)
def test_special_case():
    pass
```

### Accessing VCR information

```python
@pytest.mark.vcr
def test_with_vcr_info(vcr):
    # Make API call
    result = my_llm_function()

    # Check if we're using a recording
    if vcr.play_count > 0:
        print("Using recorded response")
    else:
        print("Making live API call")
```

### Filtering dynamic data

For non-deterministic data like timestamps:

```python
@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [("x-timestamp", "FILTERED")],
        "before_record_response": scrub_timestamps,
    }
```

## Troubleshooting

### "Cassette not found" error

The test is trying to replay a cassette that doesn't exist. Run with `--record-mode=once` to create it.

### "Request not found in cassette" error

The test is making a request that wasn't recorded. Either:
1. Re-record the cassette with `--record-mode=rewrite`
2. Use `--record-mode=new_episodes` to append new requests

### Cassettes are too large

Consider:
1. Filtering unnecessary response data
2. Splitting tests into smaller units
3. Using compression (cassettes are YAML, so they compress well)

## Example Test Structure

```python
# tests/test_features.py
import pytest
from promptic import llm
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    score: float

@pytest.mark.vcr
def test_sentiment_analysis():
    """Test that will use VCR for API calls."""

    @llm(model="gpt-4o-mini", temperature=0)
    def analyze_sentiment(text: str) -> Analysis:
        """Analyze the sentiment of: {text}"""

    result = analyze_sentiment("I love this library!")
    assert result.sentiment == "positive"
    assert result.score > 0.8

@pytest.mark.no_vcr
def test_data_validation():
    """Test that doesn't need VCR (no API calls)."""

    # Pure Python logic testing
    analysis = Analysis(sentiment="positive", score=0.9)
    assert analysis.sentiment in ["positive", "negative", "neutral"]
```

## CI/CD Integration

In your CI/CD pipeline:

```yaml
# .github/workflows/test.yml
- name: Run tests with VCR
  run: |
    # Use 'none' mode to ensure no accidental API calls
    uv run pytest --record-mode=none
```

This ensures tests only use recorded cassettes and never make live API calls in CI.
