#!/bin/bash

# This script sets up a conda environment for the Circulation Benchmarking project.

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the name of the conda environment
ENV_NAME="circ-bench"

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Miniconda or Anaconda."
    echo "Visit https://docs.conda.io/en/latest/miniconda.html for installation instructions."
    exit 1
fi

echo "Found conda installation."

# Check if the environment already exists
if conda env list | grep -q -w "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
    read -p "Do you want to remove the existing environment and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME"
        echo "Creating new environment from environment.yml..."
        conda env create -f environment.yml
    else
        echo "Skipping environment creation. To run the scripts, please activate the existing environment:"
        echo "conda activate $ENV_NAME"
        exit 0
    fi
else
    echo "Creating conda environment '$ENV_NAME' from environment.yml..."
    conda env create -f environment.yml
fi

echo ""
echo "========================================================================"
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "After activating, you can run the main script:"
echo "  ./scripts/run_scripts.sh /path/to/your/data"
echo "========================================================================"
