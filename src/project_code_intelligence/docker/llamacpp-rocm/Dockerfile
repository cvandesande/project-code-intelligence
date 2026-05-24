ARG ROCM_BASE_IMAGE=debian:stable-slim
FROM ${ROCM_BASE_IMAGE}

RUN set -eu; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl libatomic1 libstdc++6 python3 unzip; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/project-code-intelligence
COPY src ./src
COPY scripts/select_llamacpp_rocm_bundle.py ./scripts/select_llamacpp_rocm_bundle.py
COPY pci-embedding-server ./pci-embedding-server
COPY docker/llamacpp-rocm/entrypoint.sh /usr/local/bin/project-code-intelligence-llamacpp-rocm

RUN chmod +x ./scripts/select_llamacpp_rocm_bundle.py \
    ./pci-embedding-server \
    /usr/local/bin/project-code-intelligence-llamacpp-rocm

ENV PYTHONPATH=/opt/project-code-intelligence/src

ENTRYPOINT ["project-code-intelligence-llamacpp-rocm"]
