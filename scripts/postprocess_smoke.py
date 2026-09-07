#!/usr/bin/env python3
"""CPU runtime gate: real SPAN weights, framewise upscale and mosaic fallback."""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types
import urllib.request
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    comfy_dir = args.comfy_dir.resolve()
    os.chdir(comfy_dir)
    sys.path.insert(0, str(comfy_dir))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import folder_paths
    from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
    package = comfy_dir / "custom_nodes/ComfyUI-WanLoopBatch"
    post = module_at("wan_post_smoke", package / "postprocess_nodes.py")
    mosaic = module_at("wan_mosaic_smoke", package / "mosaic_nodes.py")
    manifest = json.loads((ROOT / "config/wan22-models.json").read_text())
    asset = next(m for m in manifest["models"] if m["path"].endswith("2xNomosUni_span_multijpg.safetensors"))
    with tempfile.TemporaryDirectory(prefix="wan-span-smoke-") as temporary:
        filename = pathlib.PurePosixPath(asset["path"]).name
        target = pathlib.Path(temporary) / filename
        # Only the 4.5 MB upscaler, never WAN checkpoints. Removed after test.
        with urllib.request.urlopen(asset["url"], timeout=60) as response:
            content = response.read(asset["size_bytes"] + 1)
        assert len(content) == asset["size_bytes"]
        assert hashlib.sha256(content).hexdigest() == asset["sha256"]
        target.write_bytes(content)
        folder_paths.add_model_folder_path("upscale_models", temporary)
        model = UpscaleModelLoader().load_model(filename)[0]
        assert model.scale == 2
        image = torch.rand((2, 32, 32, 3), generator=torch.Generator().manual_seed(17))
        with torch.inference_mode():
            output = post.WanLoopModelUpscale().upscale(model, image, 2.0)[0]
        assert output.shape == (2, 64, 64, 3)
        assert output.device.type == "cpu" and torch.isfinite(output).all()
        assert output.min() >= 0 and output.max() <= 1
        print("[post-smoke] real SPAN load + framewise 2x forward OK (CPU)")

        calls = []
        def fake_four_x(_model, batch):
            calls.append(len(batch))
            return (batch.repeat_interleave(4, 1).repeat_interleave(4, 2),)
        with patch.object(ImageUpscaleWithModel, "upscale", side_effect=fake_four_x):
            alternate = post.WanLoopModelUpscale().upscale(types.SimpleNamespace(scale=4), image, 2.0)[0]
        assert calls == [1, 1] and alternate.shape == output.shape
        print("[post-smoke] alternate 4x model retains only net-2x frames OK")

        class FakeDetector:
            names = {0: "pussy", 1: "anus"}
            def __init__(self):
                self.calls = []
                self.offloads = []
            def predict(self, **kwargs):
                self.calls.append(kwargs["device"])
                assert kwargs["classes"] == [0] and kwargs["half"] is False
                if kwargs["device"] != "cpu":
                    raise torch.cuda.OutOfMemoryError("simulated detector OOM")
                return [types.SimpleNamespace(masks=None, boxes=None)]
            def to(self, device):
                self.offloads.append(device)
                return self
        detector = FakeDetector()
        with patch.object(mosaic, "_load_model", return_value=detector), \
             patch.object(mosaic, "_inference_device", return_value="cuda:0"):
            safe = mosaic.WanAutoMosaicVideo().apply(
                image, mosaic.MODEL_FILENAME, "JUST", .3, .5, 0, 3, "pussy", "auto")[0]
        torch.testing.assert_close(safe, image)
        assert detector.calls == ["cuda:0", "cpu", "cpu"]
        assert detector.offloads == ["cpu"]
        assert mosaic._inference_device("cpu") == "cpu"
        print("[post-smoke] every-frame mosaic / GPU OOM -> CPU retry OK (simulated OOM)")

        # Fail closed and offload even when inference fails for another reason.
        failed = FakeDetector()
        with patch.object(mosaic, "_load_model", return_value=failed), \
             patch.object(mosaic, "_inference_device", return_value="cuda:0"), \
             patch.object(failed, "predict", side_effect=RuntimeError("simulated failure")):
            try:
                mosaic.WanAutoMosaicVideo().apply(
                    image, mosaic.MODEL_FILENAME, "JUST", .3, .5, 0, 3, "pussy", "auto")
            except RuntimeError as exc:
                assert str(exc) == "simulated failure"
            else:
                raise AssertionError("Detector failure was silently ignored")
        assert failed.offloads == ["cpu"]
        print("[post-smoke] detector failure propagates + releases GPU model OK")


if __name__ == "__main__":
    main()
