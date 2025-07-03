# Migrating Existing Tests to VCR

This guide explains how to migrate existing promptic tests to use VCR.py for recording and replaying API interactions.

## Why Migrate to VCR?

- **Speed**: Tests run 10-100x faster using recordings instead of live API calls
- **Cost**: Eliminate API costs during test runs
- **Reliability**: Tests work offline and aren't affected by API rate limits or outages
- **CI/CD**: Tests can run in environments without API keys

## Migration Steps

### 1. Add the VCR marker to existing tests

Simply add `@pytest.mark.vcr` to any test that makes LLM API calls:

```python
# Before
def test_translation():
    @llm(model="gpt-4o-mini", temperature=0)
    def translate(text: str, language: str) -> str:
        """Translate {text} to {language}"""

    result = translate("Hello", "Spanish")
    assert "Hola" in result

# After
@pytest.mark.vcr  # Add this line
def test_translation():
    @llm(model="gpt-4o-mini", temperature=0)
    def translate(text: str, language: str) -> str:
        """Translate {text} to {language}"""

    result = translate("Hello", "Spanish")
    assert "Hola" in result
```

### 2. Record initial cassettes

Run tests with recording enabled to create cassettes:

```bash
# Record specific test
uv run pytest tests/test_module.py::test_translation --record-mode=once

# Record all tests in a module
uv run pytest tests/test_module.py --record-mode=once

# Record all tests
uv run pytest --record-mode=once
```

### 3. Verify cassettes work

Run tests in replay-only mode to ensure cassettes are working:

```bash
uv run pytest tests/test_module.py --record-mode=none
```

### 4. Commit cassettes to version control

```bash
git add tests/cassettes/
git commit -m "Add VCR cassettes for test recordings"
```

## Best Practices for Migration

### 1. Ensure deterministic tests

Make sure tests use `temperature=0` for consistent results:

```python
@pytest.mark.vcr
def test_deterministic():
    @llm(model="gpt-4o-mini", temperature=0)  # Always use temperature=0
    def generate() -> str:
        """Generate a greeting"""
```

### 2. Group related tests

Tests that share similar prompts can be grouped in the same test file to share cassette directories:

```python
# tests/test_translations.py
@pytest.mark.vcr
def test_translate_spanish():
    # ...

@pytest.mark.vcr
def test_translate_french():
    # ...
```

### 3. Handle model-specific tests

For tests that use multiple models, consider parameterization:

```python
@pytest.mark.parametrize("model", ["gpt-4o-mini", "claude-3-haiku-20240307"])
@pytest.mark.vcr
def test_multiple_models(model):
    @llm(model=model, temperature=0)
    def generate() -> str:
        """Say hello"""

    result = generate()
    assert len(result) > 0
```

### 4. Update CI/CD configuration

Update your CI workflow to use replay-only mode:

```yaml
# .github/workflows/tests.yml
- name: Run tests
  run: uv run pytest --record-mode=none
```

## Handling Special Cases

### Tests with dynamic data

For tests that use timestamps or random data, ensure the test logic accounts for recorded responses:

```python
@pytest.mark.vcr
def test_with_timestamp():
    @llm(model="gpt-4o-mini", temperature=0)
    def get_info() -> str:
        """Get some information"""

    result = get_info()
    # Don't assert on dynamic content that might change
    assert isinstance(result, str)
    assert len(result) > 0
```

### Tests that shouldn't be recorded

Some tests might not be suitable for VCR (e.g., testing rate limits, streaming):

```python
@pytest.mark.no_vcr  # Skip VCR for this test
def test_rate_limiting():
    # Test actual rate limiting behavior
    pass
```

### Updating cassettes

When prompts or expected outputs change, re-record affected cassettes:

```bash
# Re-record specific test
uv run pytest tests/test_module.py::test_name --record-mode=rewrite

# Use the just command
just update-cassette test_name
```

## Troubleshooting

### "Cassette not found" error

The test is looking for a cassette that doesn't exist. Record it:

```bash
uv run pytest path/to/test.py::test_name --record-mode=once
```

### "Request not found in cassette" error

The test is making a different request than what was recorded. Either:
1. The prompt or parameters changed - re-record with `--record-mode=rewrite`
2. The test has non-deterministic behavior - fix the test to be deterministic

### Large cassette files

If cassettes are too large:
1. Consider splitting tests into smaller units
2. Use `.gitattributes` to mark them as binary:
   ```
   tests/cassettes/**/*.yaml binary
   ```

## Example Migration

Here's a complete example of migrating a test file:

```python
# Original test file
from promptic import llm
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    score: float

def test_sentiment_analysis():
    @llm(model="gpt-4o-mini")  # No temperature set
    def analyze(text: str) -> Analysis:
        """Analyze sentiment of: {text}"""

    result = analyze("I love this!")
    assert result.sentiment == "positive"

# Migrated test file
import pytest
from promptic import llm
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    score: float

@pytest.mark.vcr  # Added VCR marker
def test_sentiment_analysis():
    @llm(model="gpt-4o-mini", temperature=0)  # Added temperature=0
    def analyze(text: str) -> Analysis:
        """Analyze sentiment of: {text}"""

    result = analyze("I love this!")
    assert result.sentiment == "positive"
    assert result.score > 0.8  # More specific assertion
```

After migration, record the cassette:
```bash
uv run pytest tests/test_sentiment.py --record-mode=once
```

The test will now run using the recorded response, making it fast and reliable!
