#!/bin/sh
set -eu

LLAMACPP_DIR=/opt/llamacpp-rocm
PROJECT_DIR=/opt/project-code-intelligence

: "${PCI_HF_MODEL_REPO:=Qwen/Qwen3-Embedding-0.6B-GGUF}"
: "${PCI_HF_MODEL_FILE:=Qwen3-Embedding-0.6B-Q8_0.gguf}"
: "${PCI_LLAMA_MODEL:=/models/$PCI_HF_MODEL_FILE}"

mkdir -p "$LLAMACPP_DIR" /models

if [ -z "${PCI_LLAMA_CPP_ROCM_URL:-}" ]; then
	eval "$("$PROJECT_DIR/scripts/select_llamacpp_rocm_bundle.py" --format env)"
fi

if [ -z "${PCI_LLAMA_CPP_ROCM_URL:-}" ]; then
	echo "PCI_LLAMA_CPP_ROCM_URL was not set or detected" >&2
	exit 1
fi

: "${PCI_LLAMA_CPP_ROCM_ASSET:=$(basename "$PCI_LLAMA_CPP_ROCM_URL")}"
: "${PCI_LLAMA_CPP_ROCM_RELEASE:=manual}"
: "${PCI_LLAMA_CPP_ROCM_BUNDLE:=manual}"

ZIP_PATH="$LLAMACPP_DIR/$PCI_LLAMA_CPP_ROCM_ASSET"
INSTALL_DIR="$LLAMACPP_DIR/$PCI_LLAMA_CPP_ROCM_RELEASE-$PCI_LLAMA_CPP_ROCM_BUNDLE"

if [ ! -s "$ZIP_PATH" ]; then
	echo "Downloading llama.cpp ROCm bundle: $PCI_LLAMA_CPP_ROCM_URL" >&2
	curl -fL --retry 3 --retry-delay 2 -o "$ZIP_PATH" "$PCI_LLAMA_CPP_ROCM_URL"
fi

if [ ! -d "$INSTALL_DIR" ] || ! find "$INSTALL_DIR" -type f -name llama-server | grep -q .; then
	rm -rf "$INSTALL_DIR"
	mkdir -p "$INSTALL_DIR"
	unzip -q "$ZIP_PATH" -d "$INSTALL_DIR"
fi

PCI_LLAMA_SERVER="$(
	find "$INSTALL_DIR" -type f -name llama-server | sort | head -n 1
)"

if [ -z "$PCI_LLAMA_SERVER" ]; then
	echo "llama-server was not found in $INSTALL_DIR" >&2
	exit 1
fi

chmod +x "$PCI_LLAMA_SERVER"

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

PCI_LLAMA_CPP_DIR="$(dirname "$PCI_LLAMA_SERVER")"
export PCI_LLAMA_CPP_DIR
export PCI_LLAMA_SERVER
export PCI_LLAMA_MODEL

exec "$PROJECT_DIR/pci-embedding-server"
