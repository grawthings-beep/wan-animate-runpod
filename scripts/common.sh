#!/usr/bin/env bash
set -Eeuo pipefail

find_comfyui_dir() {
  if [[ -n "${COMFYUI_DIR:-}" && -f "${COMFYUI_DIR}/main.py" ]]; then
    printf '%s\n' "${COMFYUI_DIR}"
    return 0
  fi

  for candidate in \
    /opt/ComfyUI \
    /workspace/ComfyUI \
    /workspace/comfyui \
    /comfyui \
    /ComfyUI \
    /app/ComfyUI; do
    if [[ -f "${candidate}/main.py" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  local found_main
  found_main="$(find /opt /workspace /app /comfyui /ComfyUI -maxdepth 4 -type f -name main.py 2>/dev/null | head -n 1 || true)"
  if [[ -n "${found_main}" ]]; then
    dirname "${found_main}"
    return 0
  fi

  return 1
}

find_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  return 1
}

normalize_cuda_visibility() {
  if [[ "${CUDA_NORMALIZE_VISIBLE_DEVICES:-1}" != "1" ]]; then
    echo "[cuda-bootstrap] CUDA visibility normalization disabled."
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[cuda-bootstrap] nvidia-smi is not available yet; leaving CUDA_VISIBLE_DEVICES unchanged."
    return 0
  fi

  local gpu_indices gpu_count current
  gpu_indices="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null || true)"
  gpu_count="$(printf '%s\n' "${gpu_indices}" | awk 'NF { count++ } END { print count + 0 }')"

  # RunPod sometimes exposes one /dev/nvidia device while leaving
  # CUDA_VISIBLE_DEVICES unset or set to a UUID which PyTorch cannot resolve
  # inside the container. Device 0 is the correct container-local ordinal when
  # nvidia-smi reports exactly one GPU. Never rewrite multi-GPU allocations.
  if [[ "${gpu_count}" == "1" ]]; then
    current="${CUDA_VISIBLE_DEVICES-}"
    if [[ "${current}" != "0" ]]; then
      export CUDA_VISIBLE_DEVICES=0
      echo "[cuda-bootstrap] Using CUDA_VISIBLE_DEVICES=0 for the single GPU exposed by RunPod (was: ${current:-unset})."
    else
      echo "[cuda-bootstrap] CUDA_VISIBLE_DEVICES=0 already matches the single exposed GPU."
    fi
  else
    echo "[cuda-bootstrap] Detected ${gpu_count} GPUs; leaving CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}."
  fi
}

print_cuda_environment() {
  echo "[cuda-bootstrap] CUDA environment: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset} NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES-unset}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
      --query-gpu=name,uuid,driver_version,pci.bus_id,memory.total \
      --format=csv,noheader 2>&1 || true
  fi
}
