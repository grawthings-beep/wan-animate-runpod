# RunPod Wan2.2 Animate (ComfyUI)

RunPod ComfyUI Pod template for **Wan2.2 Animate** — pose- and face-driven
character animation. Feed **one reference image** (e.g. an Anima LoRA still) plus
a **driving dance video**, and Wan2.2 Animate transfers the motion + expressions
onto your character while preserving its identity. Best open-weight option for
"make this character dance" as of mid-2026.

This image bakes only ComfyUI startup glue, custom-node install, and downloader
scripts. Model files are downloaded into `/workspace/comfyui/models` at Pod
startup so a persistent RunPod volume reuses them.

## Container Image

GitHub Actions builds on push to `main`:

```text
ghcr.io/grawthings-beep/wan-animate-runpod:cuda12.8
```

After the first successful build, set the GHCR package visibility to **Public**
(Packages -> this package -> Package settings), or RunPod needs a registry secret.

## RunPod Template

```text
Type: Pod
Compute type: Nvidia GPU
Container image: ghcr.io/grawthings-beep/wan-animate-runpod:cuda12.8
Container disk: 40 GB
Volume disk: 120 GB+        (the 14B model + encoders are large)
Volume mount path: /workspace
Expose HTTP ports: 8188
```

GPU: the default model is the **fp8 14B** build (~16 GB weights), which fits a
**24 GB GPU (RTX 4090)** with `--reserve-vram`. For headroom or the bf16 build,
use **A100 80GB / H100**.

Environment variables (see `runpod-template.env.example`):

```text
PORT=8188
LISTEN=0.0.0.0
DOWNLOAD_MODELS=1
RUN_DEP_CHECK=0
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
MODEL_MANIFEST_URL=https://raw.githubusercontent.com/grawthings-beep/wan-animate-runpod/main/config/wan-animate-models.json
COMFYUI_ARGS=--reserve-vram 2
```

All model files are public on Hugging Face, so `HF_TOKEN` is optional.

## Model Layout

Startup downloads (default set — fp8):

```text
models/diffusion_models/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors
models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
models/clip_vision/clip_vision_h.safetensors
models/vae/wan_2.1_vae.safetensors
models/loras/wan/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
```

To train/run on the **full bf16** diffusion model instead, enable the
`wan2.2_animate_14B_bf16` entry in `config/wan-animate-models.json` (big GPU only).

`lightx2v` is a step-distill LoRA that lets you generate in far fewer sampling
steps — recommended for speed; drop it from the workflow for max quality.

## Custom Nodes (installed at build)

- `ComfyUI-KJNodes`
- `comfyui_controlnet_aux` (provides the DWPose estimator for the driving video)
- `ComfyUI-VideoHelperSuite` (load/save video)

Native Wan2.2 Animate nodes ship with current ComfyUI core.

## Workflow

Wan2.2 Animate has an official **built-in ComfyUI template**. In the ComfyUI UI:
`Workflow -> Browse Templates -> Video -> Wan2.2 Animate`. Set:

```text
Diffusion model : Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors
Text encoder    : umt5_xxl_fp8_e4m3fn_scaled.safetensors
CLIP vision     : clip_vision_h.safetensors
VAE             : wan_2.1_vae.safetensors
(optional LoRA) : lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
Reference image : your character still (Anima LoRA output)
Driving video   : the dance clip to transfer motion from
```

See `workflows/README.md` for tips. A `.json` you export from a working run can
be committed under `workflows/` so it loads with the template list.

## Pipeline Fit

```text
Anima LoRA still (ComfyUI) -> Wan2.2 Animate (this repo) + dance driving video
  -> dancing clip -> auto-mosaic (SAM2 / auto-mosaic-tool) for distribution
```

## Troubleshooting

- **`CUDA unknown error` / `devices = 0` crash loop:** the host driver is older
  than the `runpod/comfyui:latest` base image's CUDA build. Redeploy on a fresh
  GPU/host, or pin `ARG BASE_IMAGE` in the `Dockerfile` to a known-good tag.
- **OOM on a 24 GB GPU:** keep the **fp8** model (default), use the `lightx2v`
  LoRA, lower resolution/frame count, and raise `--reserve-vram`.
- **Model re-downloads every boot:** ensure the volume is mounted at `/workspace`
  and models live under `/workspace/comfyui/models`.
