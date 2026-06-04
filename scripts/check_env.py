#!/usr/bin/env python3
"""Lightweight sanity check (opt-in via RUN_DEP_CHECK=1).

Reports CUDA/torch visibility and whether the expected Wan2.2 Animate model
files landed in the model root. Never hard-fails the pod; this is diagnostics.
"""
import argparse
import pathlib

EXPECTED = [
    "models/diffusion_models",
    "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "models/clip_vision/clip_vision_h.safetensors",
    "models/vae/wan_2.1_vae.safetensors",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui-dir", default="")
    ap.add_argument("--model-root", default="/workspace/comfyui")
    args = ap.parse_args()

    try:
        import torch
        print(f"[check_env] torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[check_env] device: {torch.cuda.get_device_name(0)}")
        else:
            print("[check_env] WARNING: CUDA not visible to torch (driver/image mismatch?)")
    except Exception as exc:  # noqa: BLE001
        print(f"[check_env] torch import failed: {exc}")

    root = pathlib.Path(args.model_root)
    for rel in EXPECTED:
        p = root / rel
        if p.is_dir():
            files = list(p.glob("*.safetensors"))
            print(f"[check_env] {rel}/ -> {len(files)} safetensors")
        else:
            print(f"[check_env] {rel} -> {'OK' if p.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
