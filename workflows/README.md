# Workflows

These are the **official Comfy-Org Wan2.2 Animate templates**, fetched from
[`Comfy-Org/workflow_templates`](https://github.com/Comfy-Org/workflow_templates):

| File | Use |
|------|-----|
| `wan2_2_animate_character_replace.json` | Replace a character in a driving video (auto-masked via SAM2) |
| `wan2_2_animate_full_scene.json` | Animate your reference character over the full scene |

On pod start, `scripts/start.sh` copies every `*.json` here into
`<ComfyUI>/user/default/workflows/`, so they show up under **Workflows -> Open**
(it uses `cp -n`, so a workflow you edit and save on the volume is never
clobbered).

## Required nodes / models (already wired)

These graphs need more than the base Wan models. The template installs all of it:

- **Custom nodes** (`custom_nodes.txt`): `ComfyUI-KJNodes`,
  `ComfyUI-WanAnimatePreprocess` (pose/face via vitpose + yolo),
  `ComfyUI-segment-anything-2` (auto mask).
- **Models** (`config/wan-animate-models.json`): Wan2.2 Animate 14B fp8, umt5
  text encoder, CLIP Vision H, Wan 2.1 VAE, LightX2V LoRA, **WanAnimate relight
  LoRA**, and **SAM 2.1 Hiera base+** (`models/sam2/`).
- `ComfyUI-WanAnimatePreprocess` downloads its `vitpose-l-wholebody.onnx` /
  `yolov10m.onnx` detection models itself on first run.

## Saving your own

Tweak a graph in the UI, then **Workflow -> Export** and commit the `.json` here.
It'll be installed on the next pod start alongside the bundled ones.

## Tips

- Keep the **fp8** model + `lightx2v` LoRA for 24 GB GPUs and fast iteration.
- Match the driving video's frame count / fps to your target clip length.
- Anima LoRA stills stay on-model well because Animate encodes identity from the
  reference image; if it drifts too realistic, lower any Wan realism LoRAs.
