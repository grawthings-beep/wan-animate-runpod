#!/usr/bin/env python3
"""Build deterministic RunPod-ready AIO and seamless-loop workflow files."""

import argparse
import copy
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workflows" / "source" / "WAN 2.2 Smooth Workflow v6.0.json"
OUTPUTS = {
    "aio": ROOT / "workflows" / "wan22_smooth_v6_aio_runpod.json",
    "loop": ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_runpod.json",
}

MODEL_RENAMES = {
    "SmoothMix_I2V_v2_High.safetensors": "smoothMixWan2214BI2V_i2vV20High.safetensors",
    "SmoothMix_I2V_v2_Low.safetensors": "smoothMixWan2214BI2V_i2vV20Low.safetensors",
    "SmoothMix_T2V_High_v4.safetensors": "smoothMixWan2214BI2V_t2vHighV40.safetensors",
    "SmoothMix_T2V_Low_v4.safetensors": "smoothMixWan2214BI2V_t2vLowV40.safetensors",
}


def replace_model_names(value):
    if isinstance(value, str):
        return MODEL_RENAMES.get(value, value)
    if isinstance(value, list):
        return [replace_model_names(item) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_model_names(item)
            for key, item in value.items()
            if key != "videopreview"
        }
    return value


def lora(filename, strength, enabled=False):
    return {
        "on": enabled,
        "lora": filename,
        "strength": strength,
        "strengthTwo": None,
    }


def configure_lora_node(node, entries):
    node["widgets_values"] = [
        {},
        {"type": "PowerLoraLoaderHeaderWidget"},
        *entries,
        {},
        "",
    ]
    if isinstance(node.get("size"), list) and len(node["size"]) > 1:
        node["size"][1] = max(float(node["size"][1]), 190 + 48 * len(entries))


def patch_aio(graph):
    graph = replace_model_names(graph)
    by_id = {node["id"]: node for node in graph["nodes"]}

    i2v_high = [
        lora(
            "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            3.0,
            True,
        ),
        lora(
            "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
            1.0,
        ),
        lora("SmoothXXXAnimation_High.safetensors", 0.5),
    ]
    i2v_low = [
        lora(
            "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            1.5,
            True,
        ),
        lora(
            "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
            1.0,
        ),
        lora("SmoothXXXAnimation_Low.safetensors", 0.5),
    ]
    for node_id in (201, 325):
        configure_lora_node(by_id[node_id], copy.deepcopy(i2v_high))
    for node_id in (200, 324):
        configure_lora_node(by_id[node_id], copy.deepcopy(i2v_low))

    # SmoothMix T2V v4 already has acceleration baked in. Style/animation LoRAs
    # are visible for convenience but intentionally disabled by default.
    configure_lora_node(
        by_id[109],
        [
            lora("SmoothMixAnimation_High.safetensors", 0.5),
            lora("SmoothMixStyle_High.safetensors", 0.5),
        ],
    )
    configure_lora_node(
        by_id[110],
        [
            lora("SmoothMixAnimation_Low.safetensors", 0.5),
            lora("SmoothMixStyle_Low.safetensors", 0.5),
        ],
    )

    graph.setdefault("extra", {})["runpod_bundle"] = {
        "profile": "full",
        "model_manifest": "config/wan22-models.json",
        "source": "WAN 2.2 Smooth Workflow v6.0 AIO",
    }
    return graph


def nodes_in_group(graph, group_id):
    group = next(item for item in graph["groups"] if item["id"] == group_id)
    x, y, width, height = map(float, group["bounding"])
    return [
        node
        for node in graph["nodes"]
        if x <= float(node["pos"][0]) <= x + width
        and y <= float(node["pos"][1]) <= y + height
    ]


def patch_loop(aio):
    graph = copy.deepcopy(aio)
    by_id = {node["id"]: node for node in graph["nodes"]}

    # The original AIO intentionally opens with all four top-level workflows
    # bypassed. This preset activates First2LastFrame and leaves audio disabled,
    # which avoids an audible discontinuity at the loop boundary.
    for group_id in (17, 18, 34, 36):
        for node in nodes_in_group(graph, group_id):
            node["mode"] = 0 if group_id == 36 else 4
    for node in nodes_in_group(graph, 47):
        node["mode"] = 4
    by_id[339]["mode"] = 4

    by_id[338]["title"] = "1. SELECT LOOP IMAGE (FIRST FRAME)"
    by_id[342]["title"] = "2. SELECT THE SAME IMAGE (LAST FRAME)"
    existing_note = str(by_id[323].get("widgets_values") or "")
    by_id[323]["widgets_values"] = (
        "SEAMLESS LOOP PRESET\n\n"
        "Use exactly the same source image in FIRST FRAME and LAST FRAME. "
        "Describe continuous cyclic motion and avoid cuts, entrances, exits, "
        "or irreversible actions. Generate 81 frames first; extend only after "
        "the short loop is clean.\n\n"
        + existing_note
    )
    combine = by_id[332].get("widgets_values")
    if isinstance(combine, dict):
        combine["filename_prefix"] = "Video/loops/%date:yyyy-MM-dd%/%date:hhmmss%-loop"
        combine["loop_count"] = 0

    graph.setdefault("extra", {})["runpod_bundle"]["preset"] = "seamless-loop"
    return graph


def encode(graph):
    return (
        json.dumps(graph, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
        + "\n"
    ).encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    aio = patch_aio(source)
    generated = {"aio": encode(aio), "loop": encode(patch_loop(aio))}

    changed = []
    for name, path in OUTPUTS.items():
        expected = generated[name]
        if args.check:
            if not path.is_file() or path.read_bytes() != expected:
                changed.append(str(path.relative_to(ROOT)))
        else:
            path.write_bytes(expected)
            print(f"WROTE {path.relative_to(ROOT)}")

    if changed:
        print("Generated workflows are stale: " + ", ".join(changed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
