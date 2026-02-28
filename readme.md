# Installation

## Prerequisites

- [Conda](https://docs.conda.io/) (or Miniconda/Mambaforge)
- [uv](https://docs.astral.sh/uv/) – fast Python package installer
- CUDA 12.8 (for GPU support; load via `module load cuda/12.8` on HPC)

## Install with conda + uv

Dependencies are defined in `pyproject.toml` with PyTorch CUDA 12.8 from the official index.

1. Install uv (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Run the install script:

   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. Activate the environment:

   ```bash
   conda activate ./env
   ```

## Alternative: uv venv (no conda)

To use uv's native venv instead of conda:

```bash
uv sync
uv pip install packaging wheel setuptools ninja
uv pip install flash-attn==2.8.3 --no-build-isolation
source .venv/bin/activate
```

Note: `uv sync` installs PyTorch CUDA 12.8 on Linux/Windows from `pyproject.toml`. Flash-attn must be installed separately with `--no-build-isolation` (and requires CUDA in PATH).
