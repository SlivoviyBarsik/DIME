#!/bin/bash
set -e

log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1"
}

detect_cuda_major() {
    if command -v nvcc &>/dev/null; then
        nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1
    elif [ -n "${CUDA_VERSION:-}" ]; then
        echo "${CUDA_VERSION%%.*}"
    elif [ -n "${CUDA_PATH:-}" ]; then
        basename "$CUDA_PATH" | grep -oE '[0-9]+' | head -1
    else
        echo ""
    fi
}

# On Compute Canada / Alliance clusters, load modules before running this script:
#   module load python/3.11
#   module load cuda/12.x   (or whichever version is available)

log "Creating virtual environment with Python 3.11..."
uv venv --python 3.11 .venv

log "Installing requirements from requirements.txt..."
uv pip install -r requirements.txt

CUDA_MAJOR=$(detect_cuda_major)

if [ "$CUDA_MAJOR" = "12" ]; then
    log "Detected CUDA 12 — installing JAX with cuda12_pip support..."
    uv pip install "jax[cuda12_pip]==0.4.33" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
elif [ "$CUDA_MAJOR" = "11" ]; then
    log "Detected CUDA 11 — installing JAX with cuda11_pip support..."
    uv pip install "jax[cuda11_pip]==0.4.33" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    log "No CUDA detected — installing CPU-only JAX..."
    uv pip install "jax==0.4.33"
fi

log "Installing flax, torch, orbax-checkpoint..."
uv pip install jax==0.4.33 flax==0.9.0 torch==2.4.1 orbax-checkpoint==0.6.4

log "Setup complete! Activate with: source .venv/bin/activate"
