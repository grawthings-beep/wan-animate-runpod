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

# The pinned RunPod base image normally sets this in /start.sh. This image must
# own PID 1 so it can download and validate WAN assets before starting ComfyUI,
# therefore carry the base runtime constraint contract forward explicitly.
if [[ -f /opt/comfyui-runtime-constraints.txt ]]; then
  export PIP_CONSTRAINT="${PIP_CONSTRAINT:-/opt/comfyui-runtime-constraints.txt}"
fi

# This must happen before the first process imports torch. A single RunPod GPU
# is always addressed as container-local cuda:0 even when RunPod leaves a host
# UUID (or nothing) in CUDA_VISIBLE_DEVICES.
normalize_cuda_visibility
print_cuda_environment

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/comfyui}"
MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_DIR}}"
CONFIG_DIR="${CONFIG_DIR:-/workspace/config}"
MODEL_MANIFEST="${MODEL_MANIFEST:-${CONFIG_DIR}/wan22-models.json}"
MODEL_PROFILE="${MODEL_PROFILE:-loop-quality}"
PORT="${PORT:-8188}"
LISTEN="${LISTEN:-0.0.0.0}"
export MODEL_ROOT

echo "BUNDLE REVISION: ${BUNDLE_REVISION:-unknown}"
echo "MODEL PROFILE: ${MODEL_PROFILE}"
echo "MODEL ROOT: ${MODEL_ROOT}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
# RunPod's datacenter links and SSD-backed volumes benefit from Xet's enlarged
# buffers and adaptive concurrency. Set this to 0 only on a memory-constrained
# pod; the downloader will still use the normal Xet engine.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"
# One-shot Pod downloads do not benefit from Xet's chunk cache. Keeping it
# disabled is both faster and avoids consuming another 10 GB on /workspace.
export HF_XET_CHUNK_CACHE_SIZE_BYTES="${HF_XET_CHUNK_CACHE_SIZE_BYTES:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/workspace/.cache/ultralytics}"
export YOLO_AUTOINSTALL="${YOLO_AUTOINSTALL:-false}"
export YOLO_OFFLINE="${YOLO_OFFLINE:-true}"

mkdir -p \
  "${WORKSPACE_DIR}/input" \
  "${WORKSPACE_DIR}/output" \
  "${WORKSPACE_DIR}/user/default/workflows" \
  "${MODEL_ROOT}/models/checkpoints" \
  "${MODEL_ROOT}/models/auto_mosaic" \
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
  "${TORCH_HOME}" \
  "${YOLO_CONFIG_DIR}"

# A broken RunPod host can expose the GPU through nvidia-smi while every CUDA
# call fails. Detect that before paying the time and bandwidth for 40+ GB.
if [[ "${CUDA_PREFLIGHT:-1}" == "1" ]]; then
  "${PYTHON_BIN}" /opt/runpod-wan-animate/scripts/gpu_preflight.py \
    --python "${PYTHON_BIN}" \
    --timeout "${CUDA_READY_TIMEOUT:-90}" \
    --interval "${CUDA_READY_INTERVAL:-10}"
else
  echo "WARN: GPU preflight disabled (CUDA_PREFLIGHT=${CUDA_PREFLIGHT:-0})." >&2
  # Disabling the host probe must not permit custom-node installs to silently
  # replace the pinned cu128 PyTorch/TorchVision/TorchAudio stack.
  "${PYTHON_BIN}" /opt/runpod-wan-animate/scripts/gpu_preflight.py \
    --python "${PYTHON_BIN}" \
    --stack-only
fi

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

# Workflows and user settings live on the volume. Keep the familiar filename
# on a fresh volume. If an image update changes a bundled workflow, preserve
# the existing file and install the new content under a hash-versioned name.
WORKFLOW_SRC="${WORKFLOW_SRC:-/opt/runpod-wan-animate/workflows}"
WORKFLOW_DST="${WORKFLOW_DST:-${WORKSPACE_DIR}/user/default/workflows}"
if compgen -G "${WORKFLOW_SRC}/*.json" > /dev/null 2>&1; then
  mkdir -p "${WORKFLOW_DST}"
  for workflow_source in "${WORKFLOW_SRC}"/*.json; do
    workflow_name="$(basename "${workflow_source}")"
    workflow_target="${WORKFLOW_DST}/${workflow_name}"
    if [[ ! -e "${workflow_target}" ]]; then
      cp "${workflow_source}" "${workflow_target}"
      echo "Installed bundled workflow: ${workflow_name}"
    elif ! cmp -s "${workflow_source}" "${workflow_target}"; then
      workflow_stem="${workflow_name%.json}"
      workflow_hash="$(sha256sum "${workflow_source}" | cut -c1-12)"
      workflow_versioned="${WORKFLOW_DST}/${workflow_stem}-bundle-${workflow_hash}.json"
      if [[ ! -e "${workflow_versioned}" ]]; then
        cp "${workflow_source}" "${workflow_versioned}"
        echo "Installed updated workflow: $(basename "${workflow_versioned}")"
      fi
    fi
  done
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

if [[ "${manifest_ready}" != "1" ]]; then
  # The default manifest is image-owned configuration. Refresh it atomically on
  # every boot so an existing network volume cannot pin a stale model profile.
  manifest_tmp="${MODEL_MANIFEST}.tmp"
  cp /opt/runpod-wan-animate/config/wan22-models.json "${manifest_tmp}"
  mv -f "${manifest_tmp}" "${MODEL_MANIFEST}"
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
link_runtime_asset \
  "${MODEL_ROOT}/models/rife/rife49.pth" \
  "${COMFYUI_DIR}/custom_nodes/ComfyUI_Fill-Nodes/nodes/cache/rife_models/rife49.pth"

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
    --require-cuda \
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
