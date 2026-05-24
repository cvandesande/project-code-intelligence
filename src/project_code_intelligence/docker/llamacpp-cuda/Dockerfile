ARG CUDA_BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
FROM ${CUDA_BASE_IMAGE}

USER root

RUN set -eu; \
    if command -v curl >/dev/null 2>&1; then \
      :; \
    elif command -v apt-get >/dev/null 2>&1; then \
      apt-get update; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl; \
      rm -rf /var/lib/apt/lists/*; \
    elif command -v apk >/dev/null 2>&1; then \
      apk add --no-cache ca-certificates curl; \
    else \
      echo "Unsupported CUDA base image: curl was missing and neither apt-get nor apk was found" >&2; \
      exit 1; \
    fi

WORKDIR /opt/project-code-intelligence
COPY pci-embedding-server ./pci-embedding-server
COPY docker/llamacpp-cuda/entrypoint.sh /usr/local/bin/project-code-intelligence-llamacpp-cuda

RUN chmod +x ./pci-embedding-server \
    /usr/local/bin/project-code-intelligence-llamacpp-cuda

ENTRYPOINT ["project-code-intelligence-llamacpp-cuda"]
