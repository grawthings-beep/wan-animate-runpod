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

CUSTOM_NODES_DIR="${COMFYUI_DIR}/custom_nodes"
mkdir -p "${CUSTOM_NODES_DIR}"

while IFS='|' read -r name url ref; do
  [[ -z "${name}" || "${name}" =~ ^# ]] && continue
  target="${CUSTOM_NODES_DIR}/${name}"

  if [[ ! -d "${target}/.git" ]]; then
    echo "Installing custom node ${name}"
    git clone --filter=blob:none --no-checkout "${url}" "${target}"
  else
    echo "Custom node ${name} already exists"
  fi

  if [[ -n "${ref:-}" ]]; then
    git -C "${target}" fetch --depth 1 origin "${ref}"
    git -C "${target}" checkout --detach FETCH_HEAD
  elif [[ ! -e "${target}/HEAD" ]]; then
    git -C "${target}" checkout --detach origin/HEAD
  fi

  if [[ -f "${target}/requirements.txt" ]]; then
    echo "Installing Python requirements for ${name}"
    "${PYTHON_BIN}" -m pip install -r "${target}/requirements.txt"
  elif [[ -f "${target}/requirements-no-cupy.txt" ]]; then
    # Frame Interpolation's install.py guesses CUDA from library filenames and
    # can choose the wrong CuPy wheel on newer CUDA images. RIFE itself only
    # needs the portable requirements set.
    echo "Installing portable Python requirements for ${name}"
    "${PYTHON_BIN}" -m pip install -r "${target}/requirements-no-cupy.txt"
  fi

  if [[ -f "${target}/pyproject.toml" ]]; then
    echo "Installing package ${name}"
    "${PYTHON_BIN}" -m pip install -e "${target}" || true
  fi
done < /opt/runpod-wan-animate/custom_nodes.txt
