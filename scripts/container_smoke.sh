#!/usr/bin/env bash
set -Eeuo pipefail

# Exercise the real production entrypoint without a GPU or model download.
# ComfyUI's CI quick-test imports nodes, initializes its writable user/database
# paths, parses every CLI option, then exits instead of opening port 8188.
SMOKE_ROOT="$(mktemp -d /tmp/wan-loop-smoke.XXXXXX)"
COMFYUI_DIR="${COMFYUI_DIR:-/opt/comfyui-baked}"
EXTRA_YAML="${COMFYUI_DIR}/extra_model_paths.yaml"
EXTRA_YML="${COMFYUI_DIR}/extra_model_paths.yml"

[[ -f "${EXTRA_YAML}" ]] && cp "${EXTRA_YAML}" "${SMOKE_ROOT}/extra_model_paths.yaml.bak"
[[ -f "${EXTRA_YML}" ]] && cp "${EXTRA_YML}" "${SMOKE_ROOT}/extra_model_paths.yml.bak"

cleanup() {
  if [[ -f "${SMOKE_ROOT}/extra_model_paths.yaml.bak" ]]; then
    cp "${SMOKE_ROOT}/extra_model_paths.yaml.bak" "${EXTRA_YAML}"
  else
    rm -f -- "${EXTRA_YAML}"
  fi
  if [[ -f "${SMOKE_ROOT}/extra_model_paths.yml.bak" ]]; then
    cp "${SMOKE_ROOT}/extra_model_paths.yml.bak" "${EXTRA_YML}"
  else
    rm -f -- "${EXTRA_YML}"
  fi
  rm -rf -- "${SMOKE_ROOT}"
}
trap cleanup EXIT

env \
  BOOTSTRAP_STATUS=0 \
  CUDA_PREFLIGHT=0 \
  DOWNLOAD_MODELS=0 \
  RUN_DEP_CHECK=0 \
  WORKSPACE_DIR="${SMOKE_ROOT}/workspace" \
  MODEL_ROOT="${SMOKE_ROOT}/workspace" \
  CONFIG_DIR="${SMOKE_ROOT}/config" \
  MODEL_PROFILE=loop-core \
  COMFYUI_ARGS="--cpu --quick-test-for-ci" \
  /opt/runpod-wan-animate/scripts/start.sh

test -d "${SMOKE_ROOT}/workspace/user/default/workflows"
test "$(find "${SMOKE_ROOT}/workspace/user/default/workflows" -maxdepth 1 -name '*.json' | wc -l)" -ge 10
