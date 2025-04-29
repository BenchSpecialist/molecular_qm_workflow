#!/bin/bash
# Script to manage mqc_pipeline environment with micromamba

set -e

ENV_NAME="mqc-env"
ENV_DIR="$HOME/$ENV_NAME"
ENV_FILE="$HOME/mqc_pipeline/environment.yml"

# Function to display usage information
function show_usage {
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  create   - Create a new environment from environment.yml"
    echo "  update   - Update existing environment with environment.yml"
    echo "  recreate - Remove existing environment and create a new one"
    echo "  clean    - Clean the conda package cache to free disk space"
    echo "  help     - Show this help message"
}

# Check if micromamba is available
if ! command -v micromamba &> /dev/null; then
    echo "Error: micromamba is not installed or not in PATH"
    exit 1
fi

# Process command line arguments
if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

case "$1" in
    create)
        echo "Creating environment $ENV_NAME from $ENV_FILE..."
        micromamba create --prefix $ENV_DIR --file $ENV_FILE
        ;;
    update)
        echo "Updating environment $ENV_NAME with $ENV_FILE..."
        micromamba update --prefix $ENV_DIR --file $ENV_FILE
        ;;
    recreate)
        echo "Removing environment $ENV_NAME..."
        micromamba remove --prefix $ENV_DIR --all -y
        echo "Creating environment $ENV_NAME from $ENV_FILE..."
        micromamba create --prefix $ENV_DIR --file $ENV_FILE
        ;;
    clean)
        echo "Cleaning micromamba package cache..."
        micromamba clean --all -y
        ;;
    help)
        show_usage
        ;;
    *)
        echo "Error: Unknown option: $1"
        show_usage
        exit 1
        ;;
esac

echo "Done."