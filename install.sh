#!/usr/bin/env bash
set -e

# Load CUDA module if on HPC (optional, uncomment if needed)
# module load cuda/12.8

# Create conda environment with Python 3.12
conda create --prefix ./env python=3.12 -y

# Activate and install
eval "$(conda shell.bash hook)"
conda activate ./env

# Install from pyproject.toml (PyTorch CUDA 12.8 + all deps) into conda env
uv pip install .

# flash-attn (requires CUDA toolkit in PATH; needs --no-build-isolation)
uv pip install packaging wheel setuptools ninja
uv pip install flash-attn==2.8.3 --no-build-isolation

echo "Done. Activate with: conda activate ./env"
