#!/usr/bin/env bash
set -Eeuo pipefail

source /opt/runpod-wan-animate/scripts/common.sh

COMFYUI_DIR="$(find_comfyui_dir)" || {
  echo "ERROR: could not find ComfyUI main.py. Set COMFYUI_DIR explicitly." >&2
  exit 2
}

PYTHON_BIN="$(find_python_bin)" || {
  echo "ERROR: neither python nor python3 was found in PATH." >&2
  exit 2
}

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/comfyui}"
MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_DIR}}"
CONFIG_DIR="${CONFIG_DIR:-/workspace/config}"
MODEL_MANIFEST="${MODEL_MANIFEST:-${CONFIG_DIR}/ltx-models.json}"
PORT="${PORT:-8188}"
LISTEN="${LISTEN:-0.0.0.0}"

mkdir -p "${WORKSPACE_DIR}/input" \
         "${WORKSPACE_DIR}/output" \
         "${MODEL_ROOT}/models/checkpoints" \
         "${MODEL_ROOT}/models/clip" \
         "${MODEL_ROOT}/models/clip_vision" \
         "${MODEL_ROOT}/models/configs" \
         "${MODEL_ROOT}/models/controlnet" \
         "${MODEL_ROOT}/models/diffusion_models" \
         "${MODEL_ROOT}/models/embeddings" \
         "${MODEL_ROOT}/models/loras/ltx2" \
         "${MODEL_ROOT}/models/latent_upscale_models" \
         "${MODEL_ROOT}/models/style_models" \
         "${MODEL_ROOT}/models/text_encoders" \
         "${MODEL_ROOT}/models/unet" \
         "${MODEL_ROOT}/models/upscale_models" \
         "${MODEL_ROOT}/models/vae" \
         "${MODEL_ROOT}/models/vae_approx" \
         "${CONFIG_DIR}"

write_extra_model_paths() {
  local target="$1"
  cat > "${target}" <<YAML
workspace:
  base_path: ${MODEL_ROOT}
  checkpoints: models/checkpoints/
  clip: models/clip/
  clip_vision: models/clip_vision/
  configs: models/configs/
  controlnet: models/controlnet/
  diffusion_models: models/diffusion_models/
  embeddings: models/embeddings/
  loras: models/loras/
  latent_upscale_models: models/latent_upscale_models/
  style_models: models/style_models/
  text_encoders: models/text_encoders/
  unet: models/unet/
  upscale_models: models/upscale_models/
  vae: models/vae/
  vae_approx: models/vae_approx/
YAML
}

write_extra_model_paths "${COMFYUI_DIR}/extra_model_paths.yaml"
write_extra_model_paths "${COMFYUI_DIR}/extra_model_paths.yml"

# Install bundled workflow graphs so they appear in the ComfyUI Workflows menu.
WORKFLOW_SRC="${WORKFLOW_SRC:-/opt/runpod-wan-animate/workflows}"
WORKFLOW_DST="${WORKFLOW_DST:-${COMFYUI_DIR}/user/default/workflows}"
if compgen -G "${WORKFLOW_SRC}/*.json" > /dev/null 2>&1; then
  mkdir -p "${WORKFLOW_DST}"
  # -n: never clobber a workflow the user edited and saved on the volume.
  cp -n "${WORKFLOW_SRC}"/*.json "${WORKFLOW_DST}/" 2>/dev/null || true
  echo "Installed bundled workflows into ${WORKFLOW_DST}"
fi

if [[ -n "${MODEL_MANIFEST_JSON:-}" ]]; then
  printf '%s' "${MODEL_MANIFEST_JSON}" > "${MODEL_MANIFEST}"
elif [[ -n "${MODEL_MANIFEST_URL:-}" ]]; then
  "${PYTHON_BIN}" - "${MODEL_MANIFEST_URL}" "${MODEL_MANIFEST}" <<'PY'
import pathlib
import sys
import urllib.request

url, output = sys.argv[1], pathlib.Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
request = urllib.request.Request(url, headers={"User-Agent": "runpod-wan-animate-template"})
with urllib.request.urlopen(request, timeout=60) as response:
    output.write_bytes(response.read())
PY
elif [[ ! -f "${MODEL_MANIFEST}" && -f /opt/runpod-wan-animate/config/ltx-models.json ]]; then
  cp /opt/runpod-wan-animate/config/ltx-models.json "${MODEL_MANIFEST}"
fi

# Optional: extra manifest to add more LoRAs/checkpoints without editing the base.
if [[ -n "${EXTRA_MODEL_MANIFEST_JSON:-}" ]]; then
  printf '%s' "${EXTRA_MODEL_MANIFEST_JSON}" > "${CONFIG_DIR}/extra-models.json"
elif [[ -n "${EXTRA_MODEL_MANIFEST_URL:-}" ]]; then
  "${PYTHON_BIN}" - "${EXTRA_MODEL_MANIFEST_URL}" "${CONFIG_DIR}/extra-models.json" <<'PY'
import pathlib
import sys
import urllib.request

url, output = sys.argv[1], pathlib.Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
request = urllib.request.Request(url, headers={"User-Agent": "runpod-wan-animate-template"})
with urllib.request.urlopen(request, timeout=60) as response:
    output.write_bytes(response.read())
PY
fi

if [[ "${DOWNLOAD_MODELS:-1}" == "1" && -f "${MODEL_MANIFEST}" ]]; then
  "${PYTHON_BIN}" /opt/runpod-wan-animate/scripts/download_models.py \
    --manifest "${MODEL_MANIFEST}" \
    --root "${MODEL_ROOT}"
  if [[ -f "${CONFIG_DIR}/extra-models.json" ]]; then
    "${PYTHON_BIN}" /opt/runpod-wan-animate/scripts/download_models.py \
      --manifest "${CONFIG_DIR}/extra-models.json" \
      --root "${MODEL_ROOT}"
  fi
else
  echo "Skipping model downloads."
fi

if [[ "${RUN_DEP_CHECK:-0}" == "1" ]]; then
  "${PYTHON_BIN}" /opt/runpod-wan-animate/scripts/check_env.py --comfyui-dir "${COMFYUI_DIR}" --model-root "${MODEL_ROOT}"
fi

cd "${COMFYUI_DIR}"
exec "${PYTHON_BIN}" main.py \
  --listen "${LISTEN}" \
  --port "${PORT}" \
  --enable-cors-header "${COMFYUI_CORS_ORIGIN:-*}" \
  --input-directory "${WORKSPACE_DIR}/input" \
  --output-directory "${WORKSPACE_DIR}/output" \
  ${COMFYUI_ARGS:-}
