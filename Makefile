# Variables
PIP = python -m pip

# Default target
install:
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -e .

# Install development dependencies
dev:
	$(PIP) install .[test]

# Run tests
test:
	python -m pytest

# Remove build artifacts, all __pycache__ directories.
clean:
	rm -rf dist/ *.egg-info/ build/ .eggs/
	rm -rf .pytest_cache/ .coverage
	find . -name "__pycache__" -type d -exec rm -rf {} +

