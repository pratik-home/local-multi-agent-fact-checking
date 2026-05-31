#!/usr/bin/env bash
set -euo pipefail

OLLAMA_HOME="${OLLAMA_HOME:-/home/ccl/Pratik/llm-exp/ollama}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-${OLLAMA_HOME}/models}"

OLLAMA_BIN="${OLLAMA_HOME}/bin/ollama"

if [[ ! -x "${OLLAMA_BIN}" ]]; then
  echo "Ollama binary not found at ${OLLAMA_BIN}" >&2
  exit 1
fi

mkdir -p "${OLLAMA_MODELS}"

echo "Starting Ollama server"
echo "  binary: ${OLLAMA_BIN}"
echo "  models: ${OLLAMA_MODELS}"

exec "${OLLAMA_BIN}" serve
