# RunPod quick steps

1. **Push this repo to GitHub** as `wan-animate-runpod` (main branch). GitHub
   Actions builds `ghcr.io/grawthings-beep/wan-animate-runpod:cuda12.8`.
2. **Make the GHCR package Public** (Packages -> package -> settings), so RunPod
   can pull without a registry secret.
3. **Create a RunPod Pod template:**
   - Container image: `ghcr.io/grawthings-beep/wan-animate-runpod:cuda12.8`
   - Volume mount path: `/workspace`, Volume disk `120 GB+`, Container disk `40 GB`
   - Expose HTTP port `8188`
   - Env: copy from `runpod-template.env.example` (set `MODEL_MANIFEST_URL` to your
     repo's raw config URL; `HF_TOKEN` optional).
4. **Deploy a GPU pod.** First boot downloads ~25-30 GB of models to the volume
   (one time). Watch the pod logs for `DOWNLOAD:` lines.
5. **Open ComfyUI** via RunPod Connect on port `8188`.
6. Load `Workflow -> Browse Templates -> Video -> Wan2.2 Animate`, drop in your
   reference image + driving dance video, and queue.

## Notes

- Default diffusion model is **fp8 14B** (fits 24 GB). For bf16, enable that entry
  in `config/wan-animate-models.json` and use an 80 GB GPU.
- If ComfyUI crash-loops with `CUDA unknown error`, the host driver is too old for
  the base image — redeploy on a different GPU/host, or pin `BASE_IMAGE` in the
  `Dockerfile`. See README troubleshooting.
