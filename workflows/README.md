# Workflows

Wan2.2 Animate ships an **official built-in template** in current ComfyUI, so
this folder intentionally does not bundle a hand-authored graph (a stale/invalid
`.json` would just fail to load).

## Use the built-in template

In the ComfyUI UI:

```
Workflow -> Browse Templates -> Video -> Wan2.2 Animate
```

Then point the loaders at the files this image downloads:

| Node | Value |
|------|-------|
| Load Diffusion Model | `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors` |
| Load CLIP / Text Encoder (umt5) | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| Load CLIP Vision | `clip_vision_h.safetensors` |
| Load VAE | `wan_2.1_vae.safetensors` |
| LoraLoader (optional, speed) | `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` |
| Reference image | your character still (Anima LoRA output) |
| Driving video | the dance clip (DWPose extracts pose/face automatically) |

## Saving your own

Once you have a run you like, **export it** (`Workflow -> Export`) and commit the
`.json` here. Committed workflows show up in the template/Open list on the pod.

## Tips

- Keep the **fp8** model + `lightx2v` LoRA for 24 GB GPUs and fast iteration.
- Match the driving video's frame count / fps to your target clip length.
- Anima LoRA stills stay on-model well because Animate encodes identity from the
  reference image; if it drifts too realistic, lower any Wan realism LoRAs.
