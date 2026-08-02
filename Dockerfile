# syntax=docker/dockerfile:1.7

# Pin the linux/amd64 manifest, not only the mutable Docker Hub tag. The digest
# contains the CUDA 12.8 / torch 2.10.0 runtime validated for RTX 5090.
# Override only when intentionally qualifying a new complete runtime stack.
ARG BASE_IMAGE=runpod/comfyui:1.4.4-cuda12.8@sha256:7078f94dbe28d079c487c245dc3524443e2c6225a6208a1fff8c7a652c1b3a40
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_CONSTRAINT=/opt/comfyui-runtime-constraints.txt \
    EXPECTED_TORCH_VERSION=2.10.0+cu128 \
    EXPECTED_TORCHVISION_VERSION=0.25.0+cu128 \
    EXPECTED_TORCHAUDIO_VERSION=2.10.0+cu128 \
    EXPECTED_TORCH_CUDA=12.8

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

# Keep the transfer stack isolated from ComfyUI's Python dependencies.  The
# current Hugging Face client automatically uses the Rust hf_xet backend.
RUN set -eu; \
    PYTHON_BIN="$(command -v python || command -v python3)"; \
    "${PYTHON_BIN}" -m pip install --no-cache-dir \
        --target /opt/runpod-wan-animate/downloader-libs \
        huggingface_hub==1.24.0 \
        hf-xet==1.5.2

RUN chmod +x /opt/runpod-wan-animate/scripts/*.sh \
    && /opt/runpod-wan-animate/scripts/install_custom_nodes.sh

# A custom-node requirements file must never downgrade or mix the CUDA wheel
# family. This runs without a GPU and makes an incompatible image fail in CI.
RUN "$(command -v python || command -v python3)" \
    /opt/runpod-wan-animate/scripts/gpu_preflight.py --stack-only

ARG BUNDLE_REVISION=unknown
ENV BUNDLE_REVISION=${BUNDLE_REVISION}
LABEL org.opencontainers.image.revision="${BUNDLE_REVISION}"

EXPOSE 8188

# The upstream /start.sh launches ComfyUI itself and does not exec a supplied
# CMD, so WAN intentionally owns PID 1 to perform preflight/downloads first.
# scripts/start.sh carries forward its runtime pip-constraint contract.
ENTRYPOINT []
CMD ["/opt/runpod-wan-animate/scripts/start.sh"]
