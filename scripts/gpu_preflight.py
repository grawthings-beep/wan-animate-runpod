#!/usr/bin/env python3
"""Fail fast when a RunPod host exposes NVML but cannot run CUDA."""

import argparse
import os
import subprocess
import sys
import time


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


def _result_text(result):
    output = "\n".join(
        item.strip() for item in (result.stdout or "", result.stderr or "") if item.strip()
    )
    return output[-2000:] if output else f"exit code {result.returncode}"


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
        return False, f"nvidia-smi could not run: {exc}"

    if nvidia.returncode != 0:
        return False, f"nvidia-smi failed: {_result_text(nvidia)}"

    gpu_summary = (nvidia.stdout or "").strip()
    try:
        cuda = runner(
            [python_bin, "-c", CUDA_PROBE],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"GPU visible through nvidia-smi ({gpu_summary}); CUDA probe failed: {exc}"

    if cuda.returncode != 0:
        return (
            False,
            "GPU visible through nvidia-smi "
            f"({gpu_summary}); PyTorch CUDA probe failed:\n{_result_text(cuda)}",
        )
    return True, f"nvidia-smi={gpu_summary}\ntorch_cuda={_result_text(cuda)}"


def run_preflight(python_bin, timeout_seconds, interval_seconds, runner=subprocess.run):
    deadline = time.monotonic() + max(0, timeout_seconds)
    attempt = 0
    last_diagnostic = "probe was not run"
    while True:
        attempt += 1
        ready, diagnostic = probe_once(python_bin, runner=runner)
        if ready:
            print(f"[gpu-preflight] READY attempt={attempt}")
            print(f"[gpu-preflight] {diagnostic}")
            return True

        last_diagnostic = diagnostic
        remaining = deadline - time.monotonic()
        print(
            f"[gpu-preflight] attempt={attempt} failed; "
            f"retry_budget={max(0, int(remaining))}s\n{diagnostic}",
            file=sys.stderr,
        )
        if remaining <= 0:
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

    context = {
        key: os.environ.get(key, "<unset>")
        for key in (
            "RUNPOD_POD_ID",
            "RUNPOD_DC_ID",
            "RUNPOD_POD_HOSTNAME",
            "RUNPOD_GPU_COUNT",
            "CUDA_VISIBLE_DEVICES",
        )
    }
    print(
        "[gpu-preflight] RunPod context: "
        + " ".join(f"{key}={value}" for key, value in context.items())
    )
    return 0 if run_preflight(args.python, args.timeout, args.interval) else 86


if __name__ == "__main__":
    sys.exit(main())
