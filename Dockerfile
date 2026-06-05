# syntax=docker/dockerfile:1.7

# ComfyUI base with native Wan2.2 support. Pin if :latest drifts onto a CUDA
# build newer than available RunPod host drivers (see README troubleshooting).
ARG BASE_IMAGE=runpod/comfyui:latest
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        aria2 \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY custom_nodes.txt /opt/runpod-wan-animate/custom_nodes.txt
COPY config/ /opt/runpod-wan-animate/config/
COPY scripts/ /opt/runpod-wan-animate/scripts/
COPY workflows/ /opt/runpod-wan-animate/workflows/

RUN chmod +x /opt/runpod-wan-animate/scripts/*.sh \
    && /opt/runpod-wan-animate/scripts/install_custom_nodes.sh

EXPOSE 8188

ENTRYPOINT []
CMD ["/opt/runpod-wan-animate/scripts/start.sh"]
