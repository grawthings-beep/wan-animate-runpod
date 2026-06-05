# RunPod LTX-2.3 v2v (ComfyUI)

RunPod ComfyUI Pod template for **LTX-2.3 Video-to-Video (distilled GGUF)**. Feed
a **reference image** + a **driving video** and either transfer the motion onto
your image (Motion Track: Depth / Canny / OpenPose) or edit/replace parts of the
video by prompt (Inpaint Edit). Workflow: javano2604 v1.1.2.

> Note: the repo is still named `wan-animate-runpod` for history; it now ships the
> LTX-2.3 v2v stack. The image bakes only ComfyUI startup glue, custom-node
> install, and the downloader. Models download into `/workspace/comfyui/models`
> at boot so a persistent volume reuses them.

## Container Image

GitHub Actions builds on push to `main`:

```text
ghcr.io/grawthings-beep/wan-animate-runpod:cuda12.8
```

After the first build, set the GHCR package visibility to **Public**
(Packages -> this package -> Package settings), or RunPod needs a registry secret.

## RunPod Template

```text
Type: Pod
Compute type: Nvidia GPU
Container image: ghcr.io/grawthings-beep/wan-animate-runpod:cuda12.8
Container disk: 40 GB
Volume disk: 150 GB+        (GGUF 22B + Gemma 12B + VAEs are large)
Volume mount path: /workspace
Expose HTTP ports: 8188
```

**GPU:** LTX-2.3 22B Q8 GGUF (~20 GB) + Gemma 3 12B fp8 (~12 GB) want **48 GB+
VRAM** (A100 / H100 / L40S) to avoid heavy offload. A 24 GB card runs but will be
slow from CPU offload — pick a bigger GPU on RunPod.

Environment variables (see `runpod-template.env.example`):

```text
PORT=8188
LISTEN=0.0.0.0
DOWNLOAD_MODELS=1
RUN_DEP_CHECK=0
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
MODEL_MANIFEST_URL=https://raw.githubusercontent.com/grawthings-beep/wan-animate-runpod/main/config/ltx-models.json
COMFYUI_ARGS=--reserve-vram 2
```

All model files are public on Hugging Face, so `HF_TOKEN` is optional. You can
leave `MODEL_MANIFEST_URL` unset — the image bakes `config/ltx-models.json`.

## Model Layout (auto-downloaded)

```text
models/unet/LTX-2.3-distilled-Q8_0.gguf
models/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors
models/text_encoders/ltx-2.3_text_projection_bf16.safetensors
models/vae/LTX23_video_vae_bf16.safetensors
models/vae/LTX23_audio_vae_bf16.safetensors
models/vae/taeltx2_3.safetensors                       (preview)
models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
models/loras/ltx2/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors
```

Depth/DWPose preprocessor models are auto-downloaded by `comfyui_controlnet_aux`.

## Custom Nodes (installed at build)

- `ComfyUI-LTXVideo` (LTX-2.3 sampling, IC-LoRA, AV latent, upsampler)
- `ComfyUI-KJNodes` (GGUFLoaderKJ, VAELoaderKJ, LatentUpscaleModelLoader)
- `ComfyUI-Impact-Pack` (conditional branch / switch nodes)
- `ComfyUI-Easy-Use` (loraStack, math/logic helpers)
- `comfyui_controlnet_aux` (Depth / Canny / DWPose preprocessors)
- `ComfyUI-RMBG` (background removal)
- `ComfyUI-VideoHelperSuite` (load/save video)

## Workflow

The LTX-2.3 v2v workflow is **bundled and auto-installed** on pod start — open it
from `Workflow -> Open -> ltx2.3_v2v_javano2604`. All loaders are pre-wired to the
files above. See `workflows/README.md` for the model/folder table and tips.

## Troubleshooting

- **`SageAttention` import/attn error:** the main `GGUFLoaderKJ` uses `sageattn`.
  If the base image lacks SageAttention, set that widget to `sdpa`.
- **Slow despite "distilled":** you're offloading — use a 48 GB+ GPU, and keep the
  step count low (distilled is built for few steps).
- **A loader dropdown can't see a model:** ComfyUI's `models_dir` may differ from
  the volume. Confirm the file is under `/workspace/comfyui/models/<folder>` and
  that `<folder>` is mapped in the generated `extra_model_paths.yaml`, then
  restart ComfyUI (it scans model folders at startup).
- **Model re-downloads every boot:** ensure the volume is mounted at `/workspace`
  and models live under `/workspace/comfyui/models`.
