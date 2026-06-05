# Workflows

`ltx2.3_v2v_javano2604.json` — the **LTX-2.3 Video-to-Video (distilled GGUF)**
workflow (javano2604 v1.1.2). Two modes:

- **Motion Track** — animate a still (reference image) using motion from a
  driving video, via Depth / Canny / OpenPose (default OpenPose).
- **Inpaint Edit** — add / remove / replace / restyle parts of a video by prompt.

On pod start, `scripts/start.sh` copies every `*.json` here into
`<ComfyUI>/user/default/workflows/`, so it shows up under **Workflows -> Open**
(`cp -n`, so an edit you save on the volume is never clobbered).

## Models (auto-downloaded by config/ltx-models.json)

| File | Folder |
|------|--------|
| `LTX-2.3-distilled-Q8_0.gguf` | `models/unet/` |
| `gemma_3_12B_it_fp8_scaled.safetensors` | `models/text_encoders/` |
| `ltx-2.3_text_projection_bf16.safetensors` | `models/text_encoders/` |
| `LTX23_video_vae_bf16.safetensors` | `models/vae/` |
| `LTX23_audio_vae_bf16.safetensors` | `models/vae/` |
| `taeltx2_3.safetensors` (preview) | `models/vae/` |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `models/latent_upscale_models/` |
| `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | `models/loras/ltx2/` |

The Depth/DWPose preprocessor models (`depth_anything_vitl14.pth`, `yolox_l.onnx`,
`dw-ll_ucoco_384_bs5.torchscript.pt`) are auto-downloaded by `comfyui_controlnet_aux`
on first run.

## Notes

- The main loader uses **SageAttention** (`GGUFLoaderKJ` -> `sageattn`). If the
  base image lacks SageAttention, switch that widget to `sdpa` in the node.
- Pick a big GPU on RunPod: LTX-2.3 22B Q8 (~20 GB) + Gemma 12B (~12 GB) wants
  48 GB+ VRAM to avoid heavy offload.

## Saving your own

Tweak in the UI, then **Workflow -> Export** and commit the `.json` here; it gets
installed on the next pod start alongside this one.
