# RunPod quick steps

1. **Push to GitHub** (`main`). GitHub Actions builds
   `ghcr.io/grawthings-beep/wan-animate-runpod:cuda12.8`.
2. **Make the GHCR package Public** (Packages -> package -> settings) so RunPod
   can pull without a registry secret.
3. **Create a RunPod Pod template:**
   - Container image: `ghcr.io/grawthings-beep/wan-animate-runpod:cuda12.8`
   - Volume mount path: `/workspace`, Volume disk `150 GB+`, Container disk `40 GB`
   - Expose HTTP port `8188`
   - Env: copy from `runpod-template.env.example` (`HF_TOKEN` optional;
     `MODEL_MANIFEST_URL` can be left unset — the image bakes the manifest).
4. **Deploy on a big GPU** (48 GB+ VRAM: A100 / H100 / L40S). First boot downloads
   ~35 GB of models to the volume (one time). Watch logs for `DOWNLOAD:` lines.
5. **Open ComfyUI** via RunPod Connect on port `8188`.
6. Open `Workflow -> Open -> ltx2.3_v2v_javano2604` (auto-installed on boot). Drop
   in your reference image + driving video, pick a control mode (Depth / Canny /
   OpenPose), and queue.

## Notes

- 48 GB+ VRAM strongly recommended: LTX-2.3 22B Q8 (~20 GB) + Gemma 12B (~12 GB).
  A 24 GB card offloads to CPU and gets slow.
- If the main loader errors on `sageattn`, switch `GGUFLoaderKJ` to `sdpa`.
- If a loader can't see a downloaded model, confirm it's under
  `/workspace/comfyui/models/<folder>` and restart ComfyUI (it scans at startup).
