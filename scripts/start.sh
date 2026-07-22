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
MODEL_MANIFEST="${MODEL_MANIFEST:-${CONFIG_DIR}/wan22-models.json}"
MODEL_PROFILE="${MODEL_PROFILE:-full}"
PORT="${PORT:-8188}"
LISTEN="${LISTEN:-0.0.0.0}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
# RunPod's datacenter links and SSD-backed volumes benefit from Xet's enlarged
# buffers and adaptive concurrency. Set this to 0 only on a memory-constrained
# pod; the downloader will still use the normal Xet engine.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p \
  "${WORKSPACE_DIR}/input" \
  "${WORKSPACE_DIR}/output" \
  "${WORKSPACE_DIR}/user/default/workflows" \
  "${MODEL_ROOT}/models/checkpoints" \
  "${MODEL_ROOT}/models/clip" \
  "${MODEL_ROOT}/models/clip_vision" \
  "${MODEL_ROOT}/models/diffusion_models" \
  "${MODEL_ROOT}/models/embeddings" \
  "${MODEL_ROOT}/models/loras" \
  "${MODEL_ROOT}/models/mmaudio" \
  "${MODEL_ROOT}/models/rife" \
  "${MODEL_ROOT}/models/text_encoders" \
  "${MODEL_ROOT}/models/unet" \
  "${MODEL_ROOT}/models/upscale_models" \
  "${MODEL_ROOT}/models/vae" \
  "${CONFIG_DIR}" \
  "${HF_HOME}" \
  "${HF_XET_CACHE}" \
  "${TORCH_HOME}"

write_extra_model_paths() {
  local target="$1"
  cat > "${target}" <<YAML
workspace:
  base_path: ${MODEL_ROOT}
  checkpoints: models/checkpoints/
  clip: models/clip/
  clip_vision: models/clip_vision/
  diffusion_models: models/diffusion_models/
  embeddings: models/embeddings/
  loras: models/loras/
  mmaudio: models/mmaudio/
  text_encoders: models/text_encoders/
  unet: |
    models/unet/
    models/diffusion_models/
  upscale_models: models/upscale_models/
  vae: models/vae/
YAML
}

write_extra_model_paths "${COMFYUI_DIR}/extra_model_paths.yaml"
write_extra_model_paths "${COMFYUI_DIR}/extra_model_paths.yml"

# Workflows and user settings live on the volume. Versioned filenames plus
# no-clobber copying keep user edits safe across image updates.
WORKFLOW_SRC="${WORKFLOW_SRC:-/opt/runpod-wan-animate/workflows}"
WORKFLOW_DST="${WORKFLOW_DST:-${WORKSPACE_DIR}/user/default/workflows}"
if compgen -G "${WORKFLOW_SRC}/*.json" > /dev/null 2>&1; then
  mkdir -p "${WORKFLOW_DST}"
  cp -n "${WORKFLOW_SRC}"/*.json "${WORKFLOW_DST}/" 2>/dev/null || true
  echo "Installed bundled workflows into ${WORKFLOW_DST}"
fi

manifest_ready=0
if [[ -n "${MODEL_MANIFEST_JSON:-}" ]]; then
  printf '%s' "${MODEL_MANIFEST_JSON}" > "${MODEL_MANIFEST}"
  manifest_ready=1
elif [[ -n "${MODEL_MANIFEST_URL:-}" ]]; then
  if "${PYTHON_BIN}" - "${MODEL_MANIFEST_URL}" "${MODEL_MANIFEST}" <<'PY'
import pathlib
import sys
import urllib.request

url, output = sys.argv[1], pathlib.Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
request = urllib.request.Request(url, headers={"User-Agent": "grawthings-wan22-runpod/2"})
with urllib.request.urlopen(request, timeout=60) as response:
    output.write_bytes(response.read())
PY
  then
    manifest_ready=1
  else
    echo "WARN: manifest URL failed; using the manifest baked into the image." >&2
  fi
fi

if [[ "${manifest_ready}" != "1" && ! -f "${MODEL_MANIFEST}" ]]; then
  cp /opt/runpod-wan-animate/config/wan22-models.json "${MODEL_MANIFEST}"
fi

if [[ "${DOWNLOAD_MODELS:-1}" == "1" ]]; then
  DOWNLOADER_LIBS="/opt/runpod-wan-animate/downloader-libs"
  env PYTHONPATH="${DOWNLOADER_LIBS}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" /opt/runpod-wan-animate/scripts/download_models.py \
    --manifest "${MODEL_MANIFEST}" \
    --root "${MODEL_ROOT}" \
    --profile "${MODEL_PROFILE}"
else
  echo "Skipping model downloads (DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-0})."
fi

# These two interpolation extensions look only inside their own repository.
# Link their verified volume assets so they never redownload on a new pod.
link_runtime_asset() {
  local source="$1"
  local target="$2"
  if [[ -f "${source}" ]]; then
    mkdir -p "$(dirname "${target}")"
    ln -sfn "${source}" "${target}"
  fi
}

link_runtime_asset \
  "${MODEL_ROOT}/models/rife/rife49.pth" \
  "${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/models/rife/rife49.pth"
link_runtime_asset \
  "${MODEL_ROOT}/models/rife/flownet.pkl" \
  "${COMFYUI_DIR}/custom_nodes/ComfyUI-VFI/rife/train_log/flownet.pkl"

# ComfyUI-MMAudio chooses the first mmaudio folder for its BigVGAN snapshot.
# Ensure the base-model directory points at the persistent copy as well.
mkdir -p "${COMFYUI_DIR}/models/mmaudio/nvidia"
if [[ "${COMFYUI_DIR}" != "${MODEL_ROOT}" && -d "${MODEL_ROOT}/models/mmaudio/nvidia/bigvgan_v2_44khz_128band_512x" ]]; then
  ln -sfn \
    "${MODEL_ROOT}/models/mmaudio/nvidia/bigvgan_v2_44khz_128band_512x" \
    "${COMFYUI_DIR}/models/mmaudio/nvidia/bigvgan_v2_44khz_128band_512x"
fi

if [[ "${RUN_DEP_CHECK:-1}" == "1" ]]; then
  CHECK_ARGS=()
  if [[ "${DOWNLOAD_MODELS:-1}" == "1" ]]; then
    CHECK_ARGS=(--strict)
  fi
  "${PYTHON_BIN}" /opt/runpod-wan-animate/scripts/check_env.py \
    --manifest "${MODEL_MANIFEST}" \
    --profile "${MODEL_PROFILE}" \
    --model-root "${MODEL_ROOT}" \
    "${CHECK_ARGS[@]}"
fi

read -r -a EXTRA_ARGS <<< "${COMFYUI_ARGS:---reserve-vram 3}"
CORS_ARGS=()
if [[ -n "${COMFYUI_CORS_ORIGIN:-}" ]]; then
  CORS_ARGS=(--enable-cors-header "${COMFYUI_CORS_ORIGIN}")
fi

cd "${COMFYUI_DIR}"
exec "${PYTHON_BIN}" main.py \
  --listen "${LISTEN}" \
  --port "${PORT}" \
  --input-directory "${WORKSPACE_DIR}/input" \
  --output-directory "${WORKSPACE_DIR}/output" \
  --user-directory "${WORKSPACE_DIR}/user" \
  "${CORS_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
