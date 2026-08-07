"""CPU-only automatic mosaic for completed ComfyUI video frames."""

from __future__ import annotations

import os
import pathlib
import threading

import folder_paths
import numpy as np


MODEL_FILENAME = "erax-anti-nsfw-yolo11s-v1.1.pt"
MODEL_CLASSES = ("anus", "make_love", "nipple", "penis", "vagina")
DEFAULT_CLASSES = "anus,nipple,penis,vagina"

_MODEL = None
_MODEL_PATH = None
_MODEL_LOCK = threading.Lock()


def _models_directory() -> pathlib.Path:
    model_root = os.environ.get("MODEL_ROOT", "").strip()
    if model_root:
        return pathlib.Path(model_root) / "models"
    configured = getattr(folder_paths, "models_dir", None)
    if configured:
        return pathlib.Path(configured)
    return pathlib.Path("/workspace/comfyui/models")


def _load_model(model_name: str):
    global _MODEL, _MODEL_PATH

    model_path = (_models_directory() / "auto_mosaic" / model_name).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Auto-mosaic detector is missing: {model_path}. "
            "Redeploy with DOWNLOAD_MODELS=1 and MODEL_PROFILE=loop-quality."
        )

    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_PATH != model_path:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "The bundled ultralytics runtime is unavailable. Use the "
                    "matching GHCR image instead of copying only the workflow."
                ) from exc
            _MODEL = YOLO(str(model_path), task="detect")
            _MODEL_PATH = model_path
    return _MODEL


def _selected_class_ids(names, requested: str, include_context: bool) -> list[int]:
    if isinstance(names, dict):
        normalized = {int(index): str(name).lower() for index, name in names.items()}
    else:
        normalized = {index: str(name).lower() for index, name in enumerate(names)}

    selected = {
        item.strip().lower()
        for item in str(requested).split(",")
        if item.strip()
    }
    unknown = selected.difference(MODEL_CLASSES)
    if unknown:
        raise ValueError(
            "Unknown auto-mosaic class(es): " + ", ".join(sorted(unknown))
        )
    if include_context:
        selected.add("make_love")
    ids = [index for index, name in normalized.items() if name in selected]
    if not ids:
        raise ValueError("At least one auto-mosaic target class must be selected.")
    return ids


def _expanded_box(box, width: int, height: int, expand_percent: float):
    x1, y1, x2, y2 = map(float, box)
    expand = max(0.0, float(expand_percent)) / 100.0
    grow_x = (x2 - x1) * expand / 2.0
    grow_y = (y2 - y1) * expand / 2.0
    return (
        max(0, int(np.floor(x1 - grow_x))),
        max(0, int(np.floor(y1 - grow_y))),
        min(width, int(np.ceil(x2 + grow_x))),
        min(height, int(np.ceil(y2 + grow_y))),
    )


def _boxes_to_masks(frame_boxes, height, width, expand_percent):
    masks = np.zeros((len(frame_boxes), height, width), dtype=np.bool_)
    for frame_index, boxes in enumerate(frame_boxes):
        for box in boxes:
            x1, y1, x2, y2 = _expanded_box(
                box, width, height, expand_percent
            )
            if x2 > x1 and y2 > y1:
                masks[frame_index, y1:y2, x1:x2] = True
    return masks


def _circular_temporal_union(masks, radius):
    """Bridge short detector misses, including across a seamless loop seam."""
    radius = max(0, int(radius))
    if radius == 0 or len(masks) < 2:
        return masks
    smoothed = masks.copy()
    for offset in range(1, min(radius, len(masks) - 1) + 1):
        smoothed |= np.roll(masks, offset, axis=0)
        smoothed |= np.roll(masks, -offset, axis=0)
    return smoothed


def _fixed_grid_mosaic(rgb, block_size):
    """Pixelate on a frame-global grid so moving boxes never shift the tiles."""
    block = max(2, int(block_size))
    height, width, channels = rgb.shape
    padded_height = ((height + block - 1) // block) * block
    padded_width = ((width + block - 1) // block) * block
    padded = np.pad(
        rgb,
        ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode="edge",
    )
    reduced = padded.reshape(
        padded_height // block,
        block,
        padded_width // block,
        block,
        channels,
    ).mean(axis=(1, 3), dtype=np.float32)
    return (
        reduced.repeat(block, axis=0)
        .repeat(block, axis=1)[:height, :width]
        .astype(np.uint8)
    )


class WanAutoMosaicVideo:
    """Detect explicit regions and mosaic an IMAGE batch after interpolation."""

    CATEGORY = "WAN Loop/Post Processing"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mosaicked_images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_name": ([MODEL_FILENAME],),
                "confidence": (
                    "FLOAT",
                    {"default": 0.20, "min": 0.05, "max": 0.95, "step": 0.01},
                ),
                "iou_threshold": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.01},
                ),
                "expand_percent": (
                    "FLOAT",
                    {"default": 18.0, "min": 0.0, "max": 100.0, "step": 1.0},
                ),
                "block_size": (
                    "INT",
                    {"default": 28, "min": 4, "max": 128, "step": 2},
                ),
                "temporal_radius": (
                    "INT",
                    {"default": 2, "min": 0, "max": 12, "step": 1},
                ),
                "target_classes": (
                    "STRING",
                    {"default": DEFAULT_CLASSES, "multiline": False},
                ),
                "include_make_love_context": (
                    "BOOLEAN",
                    {"default": False},
                ),
            }
        }

    def apply(
        self,
        images,
        model_name,
        confidence,
        iou_threshold,
        expand_percent,
        block_size,
        temporal_radius,
        target_classes,
        include_make_love_context,
    ):
        import torch

        if images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("Auto mosaic expects IMAGE shaped [frames, H, W, C].")

        model = _load_model(model_name)
        class_ids = _selected_class_ids(
            model.names, target_classes, include_make_love_context
        )
        frame_count, height, width, _channels = images.shape
        frame_boxes = []

        # Explicitly run on CPU. WAN and RIFE keep exclusive use of GPU VRAM.
        for frame in images:
            rgb = (
                frame[..., :3]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .clamp(0.0, 1.0)
                .numpy()
            )
            bgr = np.ascontiguousarray((rgb[:, :, ::-1] * 255.0).round().astype(np.uint8))
            result = model.predict(
                source=bgr,
                imgsz=640,
                conf=float(confidence),
                iou=float(iou_threshold),
                classes=class_ids,
                max_det=100,
                device="cpu",
                half=False,
                verbose=False,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                frame_boxes.append([])
            else:
                frame_boxes.append(result.boxes.xyxy.detach().cpu().numpy().tolist())

        masks = _boxes_to_masks(
            frame_boxes, int(height), int(width), float(expand_percent)
        )
        masks = _circular_temporal_union(masks, int(temporal_radius))

        output = torch.empty_like(images, device="cpu")
        for index, frame in enumerate(images):
            rgb = (
                frame.detach()
                .to(device="cpu", dtype=torch.float32)
                .clamp(0.0, 1.0)
                .numpy()
            )
            if masks[index].any():
                uint8_rgb = (rgb[..., :3] * 255.0).round().astype(np.uint8)
                pixelated = _fixed_grid_mosaic(uint8_rgb, block_size)
                rgb = rgb.copy()
                rgb[..., :3][masks[index]] = (
                    pixelated[masks[index]].astype(np.float32) / 255.0
                )
            output[index] = torch.from_numpy(rgb).to(dtype=images.dtype)

        return (output,)


NODE_CLASS_MAPPINGS = {"WanAutoMosaicVideo": WanAutoMosaicVideo}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAutoMosaicVideo": "WAN Auto Mosaic Completed Video (CPU)"
}
