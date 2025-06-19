.PHONY: all clean build install install-dev test

# Default Python executable
PYTHON := python3

all: install

# Clean build and other artifacts
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ *.log
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name .ruff_cache -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Install package in development mode
install: clean
	$(PYTHON) -m pip install -e .

# Install with dev dependencies
install-dev:
	$(PYTHON) -m pip install -e ".[cpu_test]"

# Build package
build: clean
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

# Run tests
test:
	pytest
