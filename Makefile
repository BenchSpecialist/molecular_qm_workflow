# Variables
PYTHON := python3
PIP = $(PYTHON) -m pip
PLATFORM := $(shell $(PYTHON) -c "import platform; print(platform.system().lower() + '-' + platform.machine().lower())")

# Default target
all: clean install

# Install the package in editable mode
# This will install the package in the current environment
install:
	$(PIP) install -e .

# Install development dependencies
dev_install:
	$(PIP) install .[dev]

# Run tests
test:
	$(PYTHON) -m pytest

# Remove build artifacts and caches
clean:
	rm -rf dist *.egg-info build .eggs
	rm -rf .pytest_cache/ .coverage *.hdf5 *.json *.pickle *.xyz
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Declare phony targets
.PHONY: all install dev_install test clean
