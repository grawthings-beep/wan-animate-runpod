"""Memory-bounded video upscaling through ComfyUI's native model runtime."""

import time


class WanLoopModelUpscale:
    CATEGORY = "WAN Loop/Post Processing"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "upscale_model": ("UPSCALE_MODEL",),
            "image": ("IMAGE",),
            "output_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
        }}

    def upscale(self, upscale_model, image, output_scale=2.0):
        import torch
        import comfy.utils
        from comfy import model_management
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

        if image.ndim != 4 or image.shape[-1] != 3 or len(image) == 0:
            raise ValueError("Upscaling expects nonempty RGB IMAGE [frames,H,W,3].")
        if not 1.0 <= float(output_scale) <= 4.0:
            raise ValueError("output_scale must be between 1 and 4.")
        start = time.perf_counter()
        count, height, width, channels = image.shape
        out_h = max(1, round(height * float(output_scale)))
        out_w = max(1, round(width * float(output_scale)))
        output = torch.empty((count, out_h, out_w, channels), device="cpu", dtype=image.dtype)
        native = ImageUpscaleWithModel()
        progress = comfy.utils.ProgressBar(count)
        # The native node supplies model-management integration, tiled inference
        # and its OOM retry. Limit its input to ONE frame, not an entire video.
        # A 4x alternate is shrunk immediately, never accumulated as a 4x batch.
        for index in range(count):
            model_management.throw_exception_if_processing_interrupted()
            frame = native.upscale(upscale_model, image[index:index + 1])[0]
            if tuple(frame.shape[1:3]) != (out_h, out_w):
                frame = comfy.utils.common_upscale(
                    frame.movedim(-1, 1), out_w, out_h, "area", "disabled"
                ).movedim(1, -1)
            output[index:index + 1].copy_(frame.detach().to(device="cpu", dtype=image.dtype))
            del frame
            progress.update(1)
        elapsed = time.perf_counter() - start
        print(f"[wan-post] upscale frames={count} input={width}x{height} "
              f"output={out_w}x{out_h} model_scale={upscale_model.scale} seconds={elapsed:.3f}")
        return (output,)


NODE_CLASS_MAPPINGS = {"WanLoopModelUpscale": WanLoopModelUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"WanLoopModelUpscale": "WAN Framewise AI Upscale (fixed output size)"}
