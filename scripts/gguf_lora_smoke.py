#!/usr/bin/env python3
"""Exercise real GGUF Q8 + synthetic LoRA arithmetic on CPU in each image.

No model download/GPU is needed. This is a runtime compatibility regression,
not a claim that any particular 14B checkpoint or style LoRA looks good.
"""

import argparse
import importlib
import importlib.util
import os
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-dir", required=True)
    args = parser.parse_args()
    comfy_dir = pathlib.Path(args.comfy_dir).resolve()
    os.chdir(comfy_dir)
    sys.path.insert(0, str(comfy_dir))
    sys.argv = [sys.argv[0], "--cpu"]

    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import comfy.lora
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType
    from gguf.quants import quantize

    assert comfy.cli_args.args.cpu, "Smoke test must never need a GPU"
    name = "wan_smoke_gguf"
    package = comfy_dir / "custom_nodes" / "ComfyUI-GGUF"
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    gguf_nodes = importlib.import_module(f"{name}.nodes")
    ops = importlib.import_module(f"{name}.ops")
    assert "UnetLoaderGGUF" in module.NODE_CLASS_MAPPINGS
    schema = module.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"].INPUT_TYPES()
    assert list(schema["required"]) == ["unet_name"]

    # A real quantized tensor, small enough to test during every Docker build.
    weights = np.linspace(-1.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    quantized = quantize(weights, GGMLQuantizationType.Q8_0)
    weight = ops.GGMLTensor(
        torch.from_numpy(quantized.copy()),
        tensor_type=GGMLQuantizationType.Q8_0,
        tensor_shape=torch.Size([32, 32]),
    )
    model = torch.nn.Module()
    model.linear = ops.GGMLOps.Linear(32, 32, bias=False)
    model.linear.load_state_dict({"weight": weight})
    inputs = torch.eye(32)
    with torch.no_grad():
        baseline = model.linear(inputs)
    original = gguf_nodes.GGUFModelPatcher(
        model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    patched = original.clone()
    up = torch.full((32, 4), 0.125)
    down = torch.full((4, 32), 0.25)
    patches = comfy.lora.load_lora({
        "test.lora_up.weight": up,
        "test.lora_down.weight": down,
        "test.alpha": torch.tensor(4.0),
    }, {"test": "linear.weight"})
    assert set(patches) == {"linear.weight"}, "Synthetic LoRA was not recognized"
    assert patched.add_patches(patches, strength_patch=0.5) == ["linear.weight"]
    cloned = patched.clone()  # Mirrors the extra clone by ModelSamplingSD3.
    assert isinstance(cloned, gguf_nodes.GGUFModelPatcher)
    cloned.patch_weight_to_device("linear.weight", device_to=torch.device("cpu"))
    with torch.no_grad():
        actual = model.linear(inputs)
    expected = baseline + torch.nn.functional.linear(inputs, 0.5 * (up @ down))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert not torch.equal(actual, baseline), "LoRA silently had no effect"
    cloned.unpatch_model()
    with torch.no_grad():
        restored = model.linear(inputs)
    torch.testing.assert_close(restored, baseline, rtol=1e-4, atol=1e-4)
    print("[gguf-smoke] Q8 + LoRA load / clone / forward / unpatch OK (CPU)")


if __name__ == "__main__":
    main()
