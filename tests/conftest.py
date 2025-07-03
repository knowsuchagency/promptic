"""Pytest configuration for VCR.py integration."""

import pytest
import os


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR for recording and replaying HTTP interactions.

    This configuration:
    - Filters out sensitive headers like API keys
    - Sets the cassette serialization format
    - Configures request matching
    - Sets up ignore patterns
    """
    return {
        # Filter sensitive headers from being recorded
        "filter_headers": [
            ("authorization", "REDACTED"),
            ("api-key", "REDACTED"),
            ("x-api-key", "REDACTED"),
            ("openai-api-key", "REDACTED"),
            ("anthropic-api-key", "REDACTED"),
            ("gemini-api-key", "REDACTED"),
        ],
        # Filter sensitive query parameters
        "filter_query_parameters": [
            ("api_key", "REDACTED"),
            ("key", "REDACTED"),
        ],
        # Match requests based on method, URI, and body
        "match_on": ["method", "uri", "body"],
        # Record mode can be overridden by --record-mode CLI option
        "record_mode": "once",
        # Cassette serialization format
        "serializer": "yaml",
        # Allow cassettes to be played even when new episodes are recorded
        "allow_playback_repeats": True,
        # Decode compressed responses
        "decode_compressed_response": True,
        # Ignore localhost for any local testing
        "ignore_hosts": ["localhost", "127.0.0.1"],
    }


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Set the directory for storing VCR cassettes.

    Cassettes are organized by test module name.
    """
    return os.path.join("tests", "cassettes", request.module.__name__.split(".")[-1])


@pytest.fixture
def vcr_cassette_name(request):
    """Generate cassette name based on test name.

    This creates descriptive cassette names that include the test function name.
    """
    return f"{request.node.name}.yaml"


# Environment variable handling for tests
@pytest.fixture(autouse=True)
def mock_api_keys_if_missing(monkeypatch):
    """Set fake API keys if they're not present in the environment.

    This allows tests to run in replay mode without real API keys.
    """
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-for-vcr-replay")
    if not os.getenv("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key-for-vcr-replay")
    if not os.getenv("GEMINI_API_KEY"):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-for-vcr-replay")


# Additional pytest configuration
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "vcr: mark test to use VCR.py for recording/replaying HTTP interactions",
    )
    config.addinivalue_line(
        "markers",
        "no_vcr: mark test to skip VCR recording (for tests that shouldn't be recorded)",
    )
