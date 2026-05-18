#!/bin/sh
set -eu

PROJECT_DIR=/opt/project-code-intelligence

: "${PCI_HF_MODEL_REPO:=Qwen/Qwen3-Embedding-0.6B-GGUF}"
: "${PCI_HF_MODEL_FILE:=Qwen3-Embedding-0.6B-Q8_0.gguf}"
: "${PCI_LLAMA_MODEL:=/models/$PCI_HF_MODEL_FILE}"
: "${PCI_LLAMA_SERVER_N_GPU_LAYERS:=999}"
: "${PCI_LLAMA_SERVER_PARALLEL:=4}"

mkdir -p /models

if [ -z "${PCI_LLAMA_SERVER:-}" ]; then
	if [ -x /app/llama-server ]; then
		PCI_LLAMA_SERVER=/app/llama-server
	else
		PCI_LLAMA_SERVER=llama-server
	fi
fi

if [ ! -s "$PCI_LLAMA_MODEL" ]; then
	if [ -z "$PCI_HF_MODEL_REPO" ] || [ -z "$PCI_HF_MODEL_FILE" ]; then
		echo "Model not found and PCI_HF_MODEL_REPO/FILE are not set: $PCI_LLAMA_MODEL" >&2
		exit 1
	fi
	model_url="https://huggingface.co/$PCI_HF_MODEL_REPO/resolve/main/$PCI_HF_MODEL_FILE?download=true"
	echo "Downloading embedding model: $PCI_HF_MODEL_REPO/$PCI_HF_MODEL_FILE" >&2
	if [ -n "${HF_TOKEN:-}" ]; then
		curl -fL --retry 3 --retry-delay 2 -H "Authorization: Bearer $HF_TOKEN" -o "$PCI_LLAMA_MODEL" "$model_url"
	else
		curl -fL --retry 3 --retry-delay 2 -o "$PCI_LLAMA_MODEL" "$model_url"
	fi
fi

export PCI_LLAMA_MODEL
export PCI_LLAMA_SERVER
export PCI_LLAMA_SERVER_N_GPU_LAYERS
export PCI_LLAMA_SERVER_PARALLEL

exec "$PROJECT_DIR/pci-embedding-server"
