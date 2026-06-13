# Contributing to QuantLite

Thank you for your interest in contributing to QuantLite. This guide covers the essentials.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/prasants/QuantLite.git
cd QuantLite

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode with all extras
pip install -e ".[dev,yahoo]"
```

## Running Tests

```bash
# Run the full test suite
python3 -m pytest tests/ -q

# Run a specific test file
python3 -m pytest tests/test_pipeline.py -v

# Run with coverage
python3 -m pytest tests/ --cov=quantlite --cov-report=term-missing
```

## Code Style

- **Linter:** We use [Ruff](https://docs.astral.sh/ruff/) for linting.
- **Line length:** 100 characters maximum.
- **Docstrings:** Google style. Every public function must have a docstring with Args, Returns, and Raises sections.
- **Type hints:** Required on all public function signatures.
- **Spelling:** British English (e.g. "optimisation", "visualisation").
- **Punctuation:** Oxford commas. No em dashes.

```bash
# Check linting
python3 -m ruff check src/ tests/

# Auto-fix where possible
python3 -m ruff check src/ tests/ --fix
```

## Submitting a Pull Request

1. Fork the repository and create a feature branch from `main`.
2. Write tests for any new functionality.
3. Ensure all tests pass: `python3 -m pytest tests/ -q`
4. Ensure linting passes: `python3 -m ruff check src/ tests/`
5. Update documentation if you add or change public APIs.
6. Open a pull request with a clear description of the change.

## Chart Standards

All visualisations must follow Stephen Few's principles:

- Maximum data-ink ratio, direct labels, no chartjunk
- Palette: `#4E79A7` primary, `#F28E2B` secondary, `#E15759` negative, `#59A14F` positive, `#76B7B2` neutral
- White background, horizontal gridlines only, 150 DPI

## Project Structure

```
src/quantlite/          # Main package
tests/                  # Test suite (mirrors package structure)
examples/               # Example scripts that generate charts
docs/                   # Documentation and images
```
