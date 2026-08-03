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

GIT_FETCH_ATTEMPTS="${GIT_FETCH_ATTEMPTS:-5}"
GIT_RETRY_DELAY_SECONDS="${GIT_RETRY_DELAY_SECONDS:-3}"

if [[ ! "${GIT_FETCH_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GIT_FETCH_ATTEMPTS must be a positive integer." >&2
  exit 2
fi
if [[ ! "${GIT_RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: GIT_RETRY_DELAY_SECONDS must be a non-negative integer." >&2
  exit 2
fi

fetch_pinned_node() {
  local name="$1"
  local url="$2"
  local ref="$3"
  local target="$4"
  local attempt status delay

  # Fetch only the pinned commit instead of cloning every remote ref. A clean
  # repository per attempt also prevents a failed HTTP transfer from poisoning
  # the next retry. HTTP/1.1 avoids intermittent HTTP/2 proxy failures seen on
  # GitHub-hosted Docker builders.
  for ((attempt = 1; attempt <= GIT_FETCH_ATTEMPTS; attempt++)); do
    rm -rf -- "${target}"
    mkdir -p "${target}"
    git -C "${target}" init --quiet
    git -C "${target}" remote add origin "${url}"

    if git -c http.version=HTTP/1.1 -C "${target}" fetch --depth 1 origin "${ref}"; then
      git -C "${target}" checkout --force --detach FETCH_HEAD
      return 0
    else
      status=$?
    fi

    if ((attempt == GIT_FETCH_ATTEMPTS)); then
      echo "ERROR: failed to fetch custom node ${name} after ${attempt} attempts." >&2
      return "${status}"
    fi

    delay=$((GIT_RETRY_DELAY_SECONDS * attempt))
    echo "WARNING: fetch failed for ${name} (attempt ${attempt}/${GIT_FETCH_ATTEMPTS}); retrying in ${delay}s." >&2
    sleep "${delay}"
  done
}

while IFS='|' read -r name url ref; do
  [[ -z "${name}" || "${name}" =~ ^# ]] && continue
  if [[ ! "${name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: unsafe custom node directory name: ${name}" >&2
    exit 2
  fi
  if [[ -z "${ref:-}" || ! "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: custom node ${name} must use a pinned 40-character commit." >&2
    exit 2
  fi
  target="${CUSTOM_NODES_DIR}/${name}"

  if [[ ! -d "${target}/.git" ]]; then
    echo "Installing custom node ${name}"
    fetch_pinned_node "${name}" "${url}" "${ref}" "${target}"
  else
    echo "Custom node ${name} already exists"
    # The image build starts clean, but keep repeated local invocations
    # deterministic as well.
    if ! git -C "${target}" cat-file -e "${ref}^{commit}" 2>/dev/null; then
      fetch_pinned_node "${name}" "${url}" "${ref}" "${target}"
    else
      git -C "${target}" checkout --force --detach "${ref}"
    fi
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
