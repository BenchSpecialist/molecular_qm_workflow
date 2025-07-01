#!/bin/bash
# Script to automatically install aimnet2calc and its dependencies

set -e

ENV_DIR="$HOME/mqc-env"
AIMNET_DIR="$HOME/aimnet2calc"

echo "Starting aimnet2calc installation process..."

# Activate the environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_DIR

# Three dependencies (pytorch, pytorch-cuda, pytorch-cluster) need separate installation
# as libmamba raises `Could not solve for environment specs` error when configuring
# in environment.yml
# CUDA Version: 12.4 for fluidstack cluster
micromamba install -y pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
micromamba install -y pytorch-cluster -c pyg
# pip install --force-reinstall --no-deps torch-cluster==1.6.3 --ignore-installed
# TORCH_VERSION='2.4.1'
# CUDA_VERSION='cu124'
# pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html

# Clone and install aimnet2calc
echo "Cloning and installing aimnet2calc..."
if [ -d "$AIMNET_DIR" ]; then
    echo "aimnet2calc directory already exists, updating..."
    cd $AIMNET_DIR
    git pull
else
    git clone https://github.com/zubatyuk/aimnet2calc.git $AIMNET_DIR
    cd $AIMNET_DIR
fi

# Install aimnet2calc
python setup.py install

echo "aimnet2calc installation complete!"