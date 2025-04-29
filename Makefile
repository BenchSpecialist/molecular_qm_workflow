.PHONY: all clean build install install-dev test

# Default Python executable
PYTHON := python3

all: install

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Install package in development mode
install: clean
	$(PYTHON) -m pip install -e .

# Install with dev dependencies
install-dev: clean
	$(PYTHON) -m pip install -e ".[dev]"

# Build package
build: clean
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

# Run tests
test:
	pytest
