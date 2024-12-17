### Requirements

Install the following tools prior to development:

- Python 3.11+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/)
- [just](https://github.com/casey/just)

### Development Setup

1. Install development dependencies:

```bash
uv sync
```

2. Install pre-commit hooks:

```bash
pre-commit install
```

3. View development recipes:

```bash
just -l
```

### Code Style and Documentation

The project uses:

- [ruff](https://github.com/astral-sh/ruff) for code formatting
- [embedme](https://github.com/zakhenry/embedme) to maintain code examples in the README

To format code and update documentation:

```bash
just format  # Format code with ruff
just embedme  # Update code examples in README
```
