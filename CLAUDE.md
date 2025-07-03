# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Promptic is a Python library for building LLM applications. It provides a lightweight abstraction over LiteLLM with an intuitive decorator-based API. The library is designed to be the "requests of LLM development" - simple, pythonic, and productive.

## Development Commands

### Testing
- **Run all tests**: `just test`
- **Run tests in parallel**: `just test-parallel`
- **Run specific test**: `just test-fn <function_name>`
- **Run tests with coverage**: `just test-cov`
- **Run a single test**: `uv run pytest tests/test_promptic.py::TestClass::test_name -xvs`

#### VCR.py Integration
Tests use VCR.py (via pytest-recording) to record and replay HTTP responses, avoiding repeated API calls:

- **Recording modes**:
  - `just test`: Run tests normally (uses existing cassettes, records new ones)
  - `just test-record`: Re-record all cassettes (uses `--record-mode=rewrite`)
  - `just test-ci`: Replay-only mode for CI (uses `--record-mode=none`)
  - `just update-cassette <pattern>`: Update specific cassettes matching pattern

- **Cassette management**:
  - Cassettes are stored in `tests/cassettes/<test_module>/<test_name>.yaml`
  - API keys are automatically filtered from recordings
  - Commit cassettes to version control for deterministic CI builds
  - Fake API keys are injected when real ones are missing (for replay mode)

### Code Quality
- **Format code**: `just format` (uses ruff)
- **Run pre-commit hooks**: `just pre-commit`
- **Update README examples**: `just embedme`

### Publishing
- **Build and publish to PyPI**: `just publish`

## Architecture

The entire library is contained in a single file: `promptic.py`. Key components:

1. **@llm decorator**: Main interface for creating LLM-powered functions. Uses function docstrings as prompt templates.

2. **Promptic class**: Core class managing LLM interactions with:
   - Conversation memory
   - Function/tool calling
   - Streaming support
   - Structured outputs with Pydantic

3. **State class**: Base class for managing conversation memory between calls.

4. **ImageBytes**: Type for handling image inputs to vision models.

## Testing Strategy

- Tests use pytest with parametrization across multiple models (GPT, Claude, Gemini)
- Two test categories: CHEAP_MODELS for basic tests, regular models for comprehensive tests
- Tests include retry logic with tenacity for handling API failures
- All new features should include tests covering both success and error cases

## Development Patterns

1. **Single Module Design**: All code is in `promptic.py` - maintain this simplicity
2. **Type Safety**: Use Pydantic for all structured outputs
3. **Model Agnostic**: Features should work across all supported LLM providers
4. **Docstring Templates**: The @llm decorator uses function docstrings as prompt templates
5. **Error Handling**: Use clear, descriptive error messages

## Common Tasks

- To add a new feature: Update `promptic.py`, add tests in `test_promptic.py`, add example in `examples/`
- To debug LLM calls: Set `PROMPTIC_DEBUG=true` environment variable
- To test with specific models: Use environment variables like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

## Dependencies

Core dependencies are minimal:
- `litellm` for LLM provider abstraction
- `pydantic` for structured outputs
- `jsonschema` for JSON schema validation

Use `uv` for all dependency management.
