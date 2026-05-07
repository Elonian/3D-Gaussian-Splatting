#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ "${SKIP_REQUIREMENTS:-0}" != "1" ]]; then
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi

if [[ "${SKIP_SIMPLE_KNN:-0}" == "1" ]]; then
    echo "Skipping simple-knn install because SKIP_SIMPLE_KNN=1."
    exit 0
fi

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "CUDA toolkit nvcc was not found, so simple-knn was not installed."
    echo "The renderer in this project does not import simple-knn, but building the extension requires CUDA_HOME."
    echo "Set CUDA_HOME to your CUDA toolkit root and rerun this script if you need simple-knn."
    exit 0
fi

python -m pip install ./sub-modules/simple-knn --no-build-isolation --verbose
