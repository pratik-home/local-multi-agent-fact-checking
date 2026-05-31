#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_HOME="${CONDA_HOME:-/home/ccl/anaconda3}"
CONDA_ENV="${CONDA_ENV:-mafc}"

export LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://localhost:11434/v1}"
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-ollama}"
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-llama3.1:8b}"

if [[ ! -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]]; then
  echo "Conda activation script not found at ${CONDA_HOME}/etc/profile.d/conda.sh" >&2
  exit 1
fi

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"

echo "Starting multi-agent fact-checking API"
echo "  repo: ${REPO_DIR}"
echo "  conda env: ${CONDA_ENV}"
echo "  local LLM: ${LOCAL_LLM_MODEL} at ${LOCAL_LLM_BASE_URL}"

exec python main.py
