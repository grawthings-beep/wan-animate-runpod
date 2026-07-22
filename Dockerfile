# syntax=docker/dockerfile:1.7

# Pinned instead of :latest so a future RunPod image cannot silently break the
# workflow/custom-node combination. Override at build time when intentionally
# testing a newer image.
ARG BASE_IMAGE=runpod/comfyui:1.4.4-cuda12.8
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        aria2 \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
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
