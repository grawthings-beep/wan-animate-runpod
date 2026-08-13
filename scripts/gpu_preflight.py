#!/usr/bin/env python3
"""Fail fast when a RunPod host exposes NVML but cannot run CUDA."""

import argparse
import csv
import io
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass


MIN_CUDA13_DRIVER = (580, 0)


TORCH_STACK_PROBE = r"""
import json
import os

import torch
import torchaudio
import torchvision

actual = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
    "torch_cuda": torch.version.cuda,
}
expected = {
    "torch": os.environ.get("EXPECTED_TORCH_VERSION", ""),
    "torchvision": os.environ.get("EXPECTED_TORCHVISION_VERSION", ""),
    "torchaudio": os.environ.get("EXPECTED_TORCHAUDIO_VERSION", ""),
    "torch_cuda": os.environ.get("EXPECTED_TORCH_CUDA", ""),
}
mismatches = {
    key: {"expected": value, "actual": actual.get(key)}
    for key, value in expected.items()
    if value and actual.get(key) != value
}
if mismatches:
    raise RuntimeError("incompatible pinned torch stack: " + json.dumps(mismatches))
print(json.dumps(actual))
"""


CUDA_PROBE = r"""
import json
import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() returned False")
if torch.cuda.device_count() < 1:
    raise RuntimeError("PyTorch reported zero CUDA devices")

device = torch.device("cuda:0")
value = torch.ones(1, device=device)
value.mul_(2)
torch.cuda.synchronize(device)
properties = torch.cuda.get_device_properties(device)
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(device),
    "vram_gib": round(properties.total_memory / (1024 ** 3), 2),
    "compute_capability": list(torch.cuda.get_device_capability(device)),
}))
"""


@dataclass(frozen=True)
class ProbeResult:
    ready: bool
    diagnostic: str
    retryable: bool = True

    # Keep two-value unpacking convenient for callers and existing tests.
    def __iter__(self):
        yield self.ready
        yield self.diagnostic


def _result_text(result):
    output = "\n".join(
        item.strip() for item in (result.stdout or "", result.stderr or "") if item.strip()
    )
    return output[-2000:] if output else f"exit code {result.returncode}"


def _version_tuple(value):
    match = re.match(r"^(\d+)\.(\d+)", value.strip())
    return tuple(map(int, match.groups())) if match else None


def _is_blackwell(name):
    return bool(
        re.search(r"RTX\s+50\d{2}", name, re.IGNORECASE)
        or re.search(
            r"\bBlackwell\b|\bB200\b|\bB300\b|\bGB200\b|RTX\s+PRO\s+\d+.*Blackwell",
            name,
            re.IGNORECASE,
        )
    )


def _incompatible_gpu_contract(gpu_summary):
    """Return a non-retryable diagnostic for an image/GPU/driver mismatch."""
    try:
        rows = list(csv.reader(io.StringIO(gpu_summary)))
    except csv.Error:
        return None

    for row in rows:
        if len(row) < 4:
            continue
        name = row[0].strip()
        driver = row[3].strip()
        is_blackwell = _is_blackwell(name)
        parsed = _version_tuple(driver)
        family = os.environ.get("WAN_GPU_FAMILY", "").strip().lower()
        expected_cuda = os.environ.get("EXPECTED_TORCH_CUDA", "").strip()

        if family == "ada" and is_blackwell:
            return (
                f"wrong image for GPU: GPU={name}; this is the Ada/CUDA 12.8 "
                "image. Deploy the loop-blackwell-cu130 image instead"
            )
        if family == "blackwell" and not is_blackwell:
            return (
                f"wrong image for GPU: GPU={name}; this is the Blackwell/CUDA "
                "13.0 image. Deploy the loop-ada-cu128 image instead"
            )
        if expected_cuda.startswith("13.") and parsed and parsed < MIN_CUDA13_DRIVER:
            minimum = ".".join(map(str, MIN_CUDA13_DRIVER))
            return (
                f"incompatible CUDA 13 driver: GPU={name} driver={driver}; "
                f"this image requires NVIDIA driver {minimum}+"
            )
    return None


def probe_torch_stack(python_bin, runner=subprocess.run):
    """Verify custom-node installs did not replace the pinned CUDA stack."""
    try:
        result = runner(
            [python_bin, "-c", TORCH_STACK_PROBE],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ProbeResult(False, f"PyTorch stack probe could not run: {exc}", False)

    if result.returncode != 0:
        return ProbeResult(
            False,
            f"PyTorch/TorchVision/TorchAudio stack validation failed:\n{_result_text(result)}",
            False,
        )
    return ProbeResult(True, _result_text(result), False)


def probe_once(python_bin, runner=subprocess.run):
    """Return (ready, diagnostic) after NVML and a real CUDA tensor operation."""
    try:
        nvidia = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ProbeResult(False, f"nvidia-smi could not run: {exc}")

    if nvidia.returncode != 0:
        return ProbeResult(False, f"nvidia-smi failed: {_result_text(nvidia)}")

    gpu_summary = (nvidia.stdout or "").strip()
    incompatible_contract = _incompatible_gpu_contract(gpu_summary)
    if incompatible_contract:
        return ProbeResult(False, incompatible_contract, False)

    try:
        cuda = runner(
            [python_bin, "-c", CUDA_PROBE],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ProbeResult(
            False,
            f"GPU visible through nvidia-smi ({gpu_summary}); CUDA probe failed: {exc}",
        )

    if cuda.returncode != 0:
        return ProbeResult(
            False,
            "GPU visible through nvidia-smi "
            f"({gpu_summary}); PyTorch CUDA probe failed:\n{_result_text(cuda)}",
        )
    return ProbeResult(
        True,
        f"nvidia-smi={gpu_summary}\ntorch_cuda={_result_text(cuda)}",
    )


def run_preflight(python_bin, timeout_seconds, interval_seconds, runner=subprocess.run):
    deadline = time.monotonic() + max(0, timeout_seconds)
    attempt = 0
    last_diagnostic = "probe was not run"
    while True:
        attempt += 1
        result = probe_once(python_bin, runner=runner)
        if result.ready:
            print(f"[gpu-preflight] READY attempt={attempt}")
            print(f"[gpu-preflight] {result.diagnostic}")
            return True

        last_diagnostic = result.diagnostic
        remaining = deadline - time.monotonic()
        print(
            f"[gpu-preflight] attempt={attempt} failed; "
            f"retry_budget={max(0, int(remaining))}s\n{result.diagnostic}",
            file=sys.stderr,
        )
        if not result.retryable or remaining <= 0:
            break
        time.sleep(min(max(0.1, interval_seconds), remaining))

    print(
        "\n[gpu-preflight] FATAL: this Pod's assigned GPU cannot execute CUDA.\n"
        "No model download was started. Terminate this Pod and deploy a new Pod "
        "so RunPod assigns a different physical host; Stop/Start can keep the Pod "
        "tied to the same host.\n"
        f"Last probe:\n{last_diagnostic}",
        file=sys.stderr,
    )
    return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--stack-only",
        action="store_true",
        help="validate the pinned torch stack without requiring a GPU",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("CUDA_READY_TIMEOUT", "90")),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.environ.get("CUDA_READY_INTERVAL", "10")),
    )
    args = parser.parse_args(argv)

    stack = probe_torch_stack(args.python)
    if not stack.ready:
        print(
            "[gpu-preflight] FATAL: incompatible PyTorch runtime.\n"
            "The image build or a custom-node dependency replaced the pinned "
            "CUDA wheel family. Do not download models with this image.\n"
            f"{stack.diagnostic}",
            file=sys.stderr,
        )
        return 87
    print(f"[gpu-preflight] TORCH STACK READY {stack.diagnostic}")
    if args.stack_only:
        return 0

    context = {
        key: os.environ.get(key, "<unset>")
        for key in (
            "RUNPOD_POD_ID",
            "RUNPOD_DC_ID",
            "RUNPOD_POD_HOSTNAME",
            "RUNPOD_GPU_COUNT",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "PIP_CONSTRAINT",
            "WAN_GPU_FAMILY",
            "EXPECTED_TORCH_CUDA",
        )
    }
    print(
        "[gpu-preflight] RunPod context: "
        + " ".join(f"{key}={value}" for key, value in context.items())
    )
    return 0 if run_preflight(args.python, args.timeout, args.interval) else 86


if __name__ == "__main__":
    sys.exit(main())
