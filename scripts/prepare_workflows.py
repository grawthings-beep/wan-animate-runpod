#!/usr/bin/env python3
"""Build deterministic RunPod-ready WAN 2.2 workflow files."""

import argparse
import copy
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOOTH_SOURCE = (
    ROOT / "workflows" / "source" / "WAN 2.2 Smooth Workflow v6.0.json"
)
LIGHTNING_SOURCE = (
    ROOT
    / "workflows"
    / "source"
    / "WAN 2.2 Native Enhanced Lightning Long Video.json"
)
OUTPUTS = {
    "aio": ROOT / "workflows" / "wan22_smooth_v6_aio_runpod.json",
    "loop": ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_runpod.json",
    "loop_core": (
        ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_core_runpod.json"
    ),
    "batch10": (
        ROOT
        / "workflows"
        / "wan22_smooth_v6_seamless_loop_batch10_runpod.json"
    ),
    "batch10_core": (
        ROOT
        / "workflows"
        / "wan22_smooth_v6_seamless_loop_batch10_core_runpod.json"
    ),
    "loop_mosaic": (
        ROOT
        / "workflows"
        / "wan22_smooth_v6_seamless_loop_auto_mosaic_runpod.json"
    ),
    "loop_mosaic_core": (
        ROOT
        / "workflows"
        / "wan22_smooth_v6_seamless_loop_core_auto_mosaic_runpod.json"
    ),
    "batch10_mosaic": (
        ROOT
        / "workflows"
        / "wan22_smooth_v6_seamless_loop_batch10_auto_mosaic_runpod.json"
    ),
    "batch10_mosaic_core": (
        ROOT
        / "workflows"
        / "wan22_smooth_v6_seamless_loop_batch10_core_auto_mosaic_runpod.json"
    ),
    "lightning": (
        ROOT
        / "workflows"
        / "wan22_native_enhanced_lightning_longvideo_runpod.json"
    ),
}

MODEL_RENAMES = {
    "SmoothMix_I2V_v2_High.safetensors": "smoothMixWan2214BI2V_i2vV20High.safetensors",
    "SmoothMix_I2V_v2_Low.safetensors": "smoothMixWan2214BI2V_i2vV20Low.safetensors",
    "SmoothMix_T2V_High_v4.safetensors": "smoothMixWan2214BI2V_t2vHighV40.safetensors",
    "SmoothMix_T2V_Low_v4.safetensors": "smoothMixWan2214BI2V_t2vLowV40.safetensors",
    (
        "wan2.2\\wan22EnhancedNSFWCameraPrompt_"
        "nsfwFASTMOVEV2FP8H.safetensors"
    ): "wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2FP8H.safetensors",
    (
        "wan2.2\\wan22EnhancedNSFWCameraPrompt_"
        "nsfwFASTMOVEV2FP8L.safetensors"
    ): "wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2FP8L.safetensors",
    (
        "wan2.2\\Wan2_2-I2V-A14B-HIGH_Q8_(lightning edition)V1.1.gguf"
    ): "wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2Q8H.gguf",
    (
        "wan2.2\\Wan2_2-I2V-A14B-LOW_Q8_(lightning edition)V1.1.gguf"
    ): "wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2Q8L.gguf",
}

SAMPLER_NODE_TYPES = {
    "KSamplerAdvanced",
    "KSamplerWithNAG (Advanced)",
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


def remove_nodes(graph, node_ids):
    """Remove nodes and every top-level link that touches them."""
    node_ids = set(node_ids)
    removed_link_ids = {
        link[0]
        for link in graph.get("links", [])
        if link[1] in node_ids or link[3] in node_ids
    }
    graph["nodes"] = [
        node for node in graph.get("nodes", []) if node["id"] not in node_ids
    ]
    graph["links"] = [
        link for link in graph.get("links", []) if link[0] not in removed_link_ids
    ]
    for node in graph["nodes"]:
        for item in node.get("inputs", []):
            if item.get("link") in removed_link_ids:
                item["link"] = None
        for item in node.get("outputs", []):
            links = item.get("links")
            if links is not None:
                item["links"] = [
                    link_id for link_id in links if link_id not in removed_link_ids
                ]


def _top_level_link(graph, link_id):
    return next(link for link in graph["links"] if link[0] == link_id)


def _append_origin_link(graph, link):
    origin = next(node for node in graph["nodes"] if node["id"] == link[1])
    output = origin["outputs"][link[2]]
    links = output.get("links")
    if links is None:
        links = []
        output["links"] = links
    links.append(link[0])


def append_link(graph, origin_id, origin_slot, target_id, target_slot, link_type):
    """Append a top-level link and update both serialized node endpoints."""
    graph["last_link_id"] += 1
    link_id = graph["last_link_id"]
    link = [
        link_id,
        origin_id,
        origin_slot,
        target_id,
        target_slot,
        link_type,
    ]
    graph["links"].append(link)
    _append_origin_link(graph, link)
    target = next(node for node in graph["nodes"] if node["id"] == target_id)
    target["inputs"][target_slot]["link"] = link_id
    return link_id


def flatten_sampler_subgraphs(graph):
    """Replace one-node sampler subgraphs with ordinary executable nodes.

    ComfyUI can bypass a subgraph by forwarding its first input, even when that
    input is MODEL and the subgraph output is LATENT. The Smooth Workflow uses
    bypassed sampler subgraphs while switching branches, so this can feed a
    ModelPatcherDynamic into VAEDecode. Flattening keeps the same sampler,
    settings, and connections while restoring type-aware node bypass behavior.
    """

    definitions = graph.get("definitions", {}).get("subgraphs", [])
    by_type = {subgraph["id"]: subgraph for subgraph in definitions}
    flattened_types = set()

    for index, instance in enumerate(list(graph["nodes"])):
        subgraph = by_type.get(instance.get("type"))
        if not subgraph or len(subgraph.get("nodes", [])) != 1:
            continue

        template = subgraph["nodes"][0]
        if template.get("type") not in SAMPLER_NODE_TYPES:
            continue

        direct = copy.deepcopy(template)
        direct["id"] = instance["id"]
        for key in ("pos", "flags", "order", "mode", "title", "color", "bgcolor", "shape"):
            if key in instance:
                direct[key] = copy.deepcopy(instance[key])

        for item in direct.get("inputs", []):
            item["link"] = None
        for item in direct.get("outputs", []):
            item["links"] = []

        internal_links = subgraph.get("links", [])
        for boundary_slot, _boundary_input in enumerate(subgraph.get("inputs", [])):
            external_link_id = instance["inputs"][boundary_slot].get("link")
            targets = [
                link
                for link in internal_links
                if link["origin_id"] == -10 and link["origin_slot"] == boundary_slot
            ]
            for target_index, target in enumerate(targets):
                if external_link_id is None:
                    continue
                if target_index == 0:
                    link_id = external_link_id
                    top_link = _top_level_link(graph, link_id)
                    top_link[3] = instance["id"]
                    top_link[4] = target["target_slot"]
                else:
                    graph["last_link_id"] += 1
                    link_id = graph["last_link_id"]
                    source = _top_level_link(graph, external_link_id)
                    top_link = [
                        link_id,
                        source[1],
                        source[2],
                        instance["id"],
                        target["target_slot"],
                        target["type"],
                    ]
                    graph["links"].append(top_link)
                    _append_origin_link(graph, top_link)
                direct["inputs"][target["target_slot"]]["link"] = link_id

        for boundary_slot, boundary_output in enumerate(subgraph.get("outputs", [])):
            sources = [
                link
                for link in internal_links
                if link["target_id"] == -20 and link["target_slot"] == boundary_slot
            ]
            if len(sources) != 1:
                raise ValueError(
                    f"sampler subgraph {subgraph['id']} output {boundary_slot} "
                    f"has {len(sources)} internal sources"
                )
            source_slot = sources[0]["origin_slot"]
            external_ids = instance["outputs"][boundary_slot].get("links") or []
            direct["outputs"][source_slot]["links"].extend(external_ids)
            for link_id in external_ids:
                top_link = _top_level_link(graph, link_id)
                top_link[1] = instance["id"]
                top_link[2] = source_slot

        graph["nodes"][index] = direct
        flattened_types.add(subgraph["id"])

    if flattened_types:
        referenced_types = {node.get("type") for node in graph["nodes"]}
        graph["definitions"]["subgraphs"] = [
            subgraph
            for subgraph in definitions
            if subgraph["id"] in referenced_types
        ]
    return graph


def patch_aio(graph):
    graph = replace_model_names(graph)
    graph = flatten_sampler_subgraphs(graph)
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


def patch_loop_model_upscale(graph):
    """Replace the loop preset's Lanczos resize with a stable model upscale.

    The 4x NMKD-Siax pass restores detail per decoded frame. A subsequent
    nearest-exact 0.5 resize preserves the workflow's existing net 2x output
    resolution without adding a large video-upscaler dependency or keeping a
    second diffusion model resident beside WAN.
    """
    by_id = {node["id"]: node for node in graph["nodes"]}
    downscale = by_id[320]
    source_link_id = downscale["inputs"][0]["link"]
    source_link = _top_level_link(graph, source_link_id)

    loader_id = graph["last_node_id"] + 1
    upscale_id = loader_id + 1
    graph["last_node_id"] = upscale_id

    loader = {
        "id": loader_id,
        "type": "UpscaleModelLoader",
        "pos": [300, 3100],
        "size": [270, 58],
        "flags": {},
        "order": 139,
        "mode": 0,
        "inputs": [
            {
                "name": "model_name",
                "type": "COMBO",
                "widget": {"name": "model_name"},
                "link": None,
            }
        ],
        "outputs": [
            {"name": "UPSCALE_MODEL", "type": "UPSCALE_MODEL", "links": []}
        ],
        "title": "AI UPSCALE MODEL (67 MB)",
        "properties": {
            "Node name for S&R": "UpscaleModelLoader",
            "cnr_id": "comfy-core",
        },
        "widgets_values": "4x_NMKD-Siax_200k.pth",
        "color": "#233",
        "bgcolor": "#355",
    }
    upscale = {
        "id": upscale_id,
        "type": "ImageUpscaleWithModel",
        "pos": [300, 3200],
        "size": [250, 72],
        "flags": {},
        "order": 140,
        "mode": 0,
        "inputs": [
            {"name": "upscale_model", "type": "UPSCALE_MODEL", "link": None},
            {"name": "image", "type": "IMAGE", "link": source_link_id},
        ],
        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
        "title": "NMKD-SIAX MODEL UPSCALE 4x",
        "properties": {
            "Node name for S&R": "ImageUpscaleWithModel",
            "cnr_id": "comfy-core",
        },
        "widgets_values": [],
        "color": "#233",
        "bgcolor": "#355",
    }
    graph["nodes"].extend([loader, upscale])

    # Preserve the decode node's existing serialized output link, changing
    # only its destination from ImageScaleBy to the model upscaler.
    source_link[3] = upscale_id
    source_link[4] = 1
    downscale["inputs"][0]["link"] = None
    append_link(graph, loader_id, 0, upscale_id, 0, "UPSCALE_MODEL")
    append_link(graph, upscale_id, 0, downscale["id"], 0, "IMAGE")

    downscale["pos"] = [580, 3200]
    downscale["order"] = 141
    downscale["title"] = "NET 2x OUTPUT (4x MODEL -> 0.5x)"
    downscale["widgets_values"] = ["nearest-exact", 0.5]
    return graph


def patch_loop(aio):
    graph = copy.deepcopy(aio)

    # The source AIO opens every top-level workflow in bypass mode. Activate
    # the retained First2LastFrame branch before pruning everything else.
    for node in nodes_in_group(graph, 36):
        node["mode"] = 0

    # This artifact is a loop-only workflow, not an AIO canvas with three
    # bypassed branches. Pruning the unused branches prevents ComfyUI from
    # reporting their optional checkpoints, LoRAs, and audio models as missing.
    unused_node_ids = set()
    for group_id in (17, 18, 34, 35, 47):
        unused_node_ids.update(node["id"] for node in nodes_in_group(graph, group_id))
    # The loop branch's GGUF loaders are disconnected alternates. Audio is
    # intentionally omitted because independently generated sound cannot loop
    # cleanly at the video boundary.
    unused_node_ids.update({308, 313, 339, 358, 359})
    remove_nodes(graph, unused_node_ids)

    keep_group_ids = {36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 48, 52}
    graph["groups"] = [
        group for group in graph.get("groups", []) if group["id"] in keep_group_ids
    ]

    by_id = {node["id"]: node for node in graph["nodes"]}
    configure_lora_node(
        by_id[325],
        [
            lora(
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
                3.0,
                True,
            ),
            lora("NSFW-22-H-e8.safetensors", 2.75, True),
            lora("SmoothXXXAnimation_High.safetensors", 1.5, True),
            lora("Cumshot_Aesthetics_High.safetensors", 1.0),
            lora("I2V_joi_trend_high.safetensors", 1.0),
            lora("Wan22_ThroatV3_High.safetensors", 1.0),
            lora(
                "cheek_bulge_fellatio_high_wan-2-2_i2v_A14B.safetensors",
                1.0,
            ),
            lora(
                "glans_licking_high_wan-2-2_i2v_A14B.safetensors",
                1.0,
            ),
            lora("head_back_high_wan-2-2_i2v_A14B.safetensors", 1.0),
            lora(
                "paizuri_unaligned_breasts_high_wan-2-2_i2v_A14B.safetensors",
                1.0,
            ),
            lora("washizukami_high_wan-2-2_i2v_A14B.safetensors", 1.0),
        ],
    )
    configure_lora_node(
        by_id[324],
        [
            lora(
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
                1.5,
                True,
            ),
            lora("NSFW-22-L-e8.safetensors", 1.65, True),
            lora("SmoothXXXAnimation_Low.safetensors", 1.0, True),
            lora("Cumshot_Aesthetics_Low.safetensors", 1.0),
            lora("I2V_joi_trend_low.safetensors", 1.0),
            lora("Wan22_ThroatV3_Low.safetensors", 1.0),
            lora("cheek_bulge_fellatio_wanvideo_i2v.safetensors", 1.0),
            lora("glans_licking_wanvideo_i2v_epoch5.safetensors", 1.0),
            lora("head_back_wanvideo_i2v_epoch5.safetensors", 1.0),
            lora(
                "paizuri_unaligned_breasts_wanvideo_i2v_epoch5.safetensors",
                1.0,
            ),
            lora("washizukami_wanvideo_i2v.safetensors", 1.0),
        ],
    )

    # The optional High/Low pairs make both loaders taller. Lift the
    # loaders into their group so they do not cover the prompt box.
    by_id[325]["pos"][1] = 2295
    by_id[324]["pos"][1] = 2295
    lora_group = next(group for group in graph["groups"] if group["id"] == 45)
    lora_group["bounding"][1] = 2255
    lora_group["bounding"][3] = 810

    resolution = by_id[328]
    resolution["properties"]["valueX"] = 528
    resolution["properties"]["valueY"] = 704
    resolution["widgets_values"] = [528, 528, 704, 704, 0, 0]

    by_id[338]["title"] = "1. SELECT LOOP IMAGE (FIRST FRAME)"
    by_id[342]["title"] = "2. SELECT THE SAME IMAGE (LAST FRAME)"
    existing_note = str(by_id[323].get("widgets_values") or "")
    by_id[323]["widgets_values"] = (
        "SEAMLESS LOOP PRESET\n\n"
        "Use exactly the same source image in FIRST FRAME and LAST FRAME. "
        "Describe continuous cyclic motion and avoid cuts, entrances, exits, "
        "or irreversible actions. Generate 81 frames first; extend only after "
        "the short loop is clean. This preset is deliberately silent so the "
        "audio track cannot introduce a seam.\n\n"
        "Core LoRAs are ON: LightX2V 3.0 High / 1.5 Low, NSFW-22 2.75 High / "
        "1.65 Low, and SmoothXXXAnimation 1.5 High / 1.0 Low. Anime Cumshot "
        "Aesthetics, JOI Handjob Trend, and Deepthroat/Face Fuck v3 High/Low "
        "pairs are available at 1.0 but OFF by default. Cumshot Aesthetics targets the official WAN base "
        "and may be unstable with an AIO/merged model. The JOI pair is native "
        "WAN 2.2 I2V-A14B. Five iroiroLoRA High/Low effect pairs are also "
        "available at 1.0 and OFF by default; enable one matching pair at a "
        "time. Default base resolution is 528 x 704.\n\n"
        + existing_note
    )
    combine = by_id[332].get("widgets_values")
    if isinstance(combine, dict):
        combine["filename_prefix"] = "Video/loops/%date:yyyy-MM-dd%/%date:hhmmss%-loop"
        combine["loop_count"] = 0

    patch_loop_model_upscale(graph)
    graph.setdefault("extra", {})["runpod_bundle"]["preset"] = "seamless-loop"
    graph["extra"]["runpod_bundle"]["profile"] = "loop-all"
    return graph


def patch_loop_core(loop):
    """Remove every disabled optional LoRA from the production core preset."""
    graph = copy.deepcopy(loop)
    for node in graph.get("nodes", []):
        if node.get("type") != "Power Lora Loader (rgthree)":
            continue
        enabled = [
            copy.deepcopy(item)
            for item in node.get("widgets_values", [])
            if isinstance(item, dict) and item.get("lora") and item.get("on")
        ]
        configure_lora_node(node, enabled)

    note = next((node for node in graph["nodes"] if node.get("id") == 323), None)
    if note:
        note["widgets_values"] = (
            "LOOP CORE PRESET\n\n"
            "Only the enabled LightX2V, NSFW-22, and SmoothXXXAnimation "
            "High/Low pairs are present. This avoids downloading 5.78 GB of "
            "disabled optional LoRAs and prevents false missing-model notices. "
            "Use the non-core workflow with MODEL_PROFILE=loop-all when those "
            "optional effects are needed.\n\n"
            + str(note.get("widgets_values") or "")
        )

    bundle = graph.setdefault("extra", {}).setdefault("runpod_bundle", {})
    bundle["preset"] = "seamless-loop-core"
    bundle["profile"] = "loop-core"
    return graph


def _loop_slot_node(node_id, slot_number, position, order):
    return {
        "id": node_id,
        "type": "WanLoopQueueSlot",
        "pos": list(position),
        "size": [520, 300],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            {
                "name": "image",
                "type": "COMBO",
                "widget": {"name": "image"},
                "link": None,
            },
            {
                "name": "positive_prompt",
                "type": "STRING",
                "widget": {"name": "positive_prompt"},
                "link": None,
            },
            {
                "name": "upload",
                "type": "IMAGEUPLOAD",
                "widget": {"name": "upload"},
                "link": None,
            },
        ],
        "outputs": [
            {
                "name": "slot",
                "type": "WAN_LOOP_QUEUE_SLOT",
                "links": [],
            }
        ],
        "properties": {
            "Node name for S&R": "WanLoopQueueSlot",
            "slot_number": slot_number,
        },
        "widgets_values": [
            "DigitalPastelLogo.png",
            "",
            "image",
        ],
        "title": f"{slot_number:02d}. IMAGE + POSITIVE PROMPT",
    }


def _loop_selector_node(node_id, position, order):
    inputs = [
        {
            "name": "active_slot",
            "type": "INT",
            "widget": {"name": "active_slot"},
            "link": None,
        },
        {
            "name": "batch_id",
            "type": "STRING",
            "widget": {"name": "batch_id"},
            "link": None,
        },
    ]
    inputs.extend(
        {
            "name": f"slot_{number:02d}",
            "type": "WAN_LOOP_QUEUE_SLOT",
            "link": None,
        }
        for number in range(1, 11)
    )
    return {
        "id": node_id,
        "type": "WanLoopQueueSelector",
        "pos": list(position),
        "size": [600, 720],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": inputs,
        "outputs": [
            {"name": "image", "type": "IMAGE", "links": []},
            {"name": "positive_prompt", "type": "STRING", "links": []},
            {"name": "filename_prefix", "type": "STRING", "links": []},
            {
                "name": "context",
                "type": "WAN_LOOP_QUEUE_CONTEXT",
                "links": [],
            },
        ],
        "properties": {"Node name for S&R": "WanLoopQueueSelector"},
        "widgets_values": [1, "increment", "click-queue-10-button"],
        "title": "BULK DROP + QUEUE 10 LOOPS — ONE JOB AT A TIME",
    }


def _loop_finalizer_node(node_id, position, order):
    return {
        "id": node_id,
        "type": "WanLoopBatchFinalize",
        "pos": list(position),
        "size": [440, 180],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            {"name": "filenames", "type": "VHS_FILENAMES", "link": None},
            {
                "name": "context",
                "type": "WAN_LOOP_QUEUE_CONTEXT",
                "link": None,
            },
            {
                "name": "expected_count",
                "type": "INT",
                "widget": {"name": "expected_count"},
                "link": None,
            },
        ],
        "outputs": [{"name": "status", "type": "STRING", "links": None}],
        "properties": {"Node name for S&R": "WanLoopBatchFinalize"},
        "widgets_values": [10],
        "title": "ZIP + AUTO DOWNLOAD AFTER #10",
    }


def patch_loop_batch10(loop):
    """Create ten independent sequential queue jobs from one loop graph."""
    graph = copy.deepcopy(loop)
    remove_nodes(graph, {333, 338, 342})

    referenced_types = {node.get("type") for node in graph["nodes"]}
    definitions = graph.get("definitions", {}).get("subgraphs", [])
    graph["definitions"]["subgraphs"] = [
        item for item in definitions if item["id"] in referenced_types
    ]

    graph["groups"] = [
        group for group in graph.get("groups", []) if group["id"] not in {39, 44, 48}
    ]
    main_group = next(group for group in graph["groups"] if group["id"] == 36)
    main_group["bounding"] = [-5240, 2299, 7060, 4150]

    first_id = graph["last_node_id"] + 1
    slot_ids = list(range(first_id, first_id + 10))
    selector_id = first_id + 10
    finalizer_id = first_id + 11
    next_order = max(node.get("order", 0) for node in graph["nodes"]) + 1

    for index, node_id in enumerate(slot_ids):
        column = index % 2
        row = index // 2
        graph["nodes"].append(
            _loop_slot_node(
                node_id,
                index + 1,
                (-5160 + column * 560, 3020 + row * 650),
                next_order + index,
            )
        )
    graph["nodes"].append(
        _loop_selector_node(selector_id, (-4020, 3020), next_order + 10)
    )
    graph["nodes"].append(
        _loop_finalizer_node(finalizer_id, (1400, 3020), next_order + 11)
    )
    graph["last_node_id"] = finalizer_id

    for index, slot_id in enumerate(slot_ids):
        append_link(
            graph,
            slot_id,
            0,
            selector_id,
            index + 2,
            "WAN_LOOP_QUEUE_SLOT",
        )

    # The selected image is reused as both ends of the seamless loop, so there
    # is no visual jump caused by re-inserting a different first frame.
    for target_id, target_slot in ((352, 1), (350, 1), (343, 5), (343, 6)):
        append_link(graph, selector_id, 0, target_id, target_slot, "IMAGE")
    append_link(graph, selector_id, 1, 305, 0, "STRING")
    append_link(graph, selector_id, 2, 332, 6, "STRING")
    append_link(
        graph,
        selector_id,
        3,
        finalizer_id,
        1,
        "WAN_LOOP_QUEUE_CONTEXT",
    )
    append_link(graph, 332, 0, finalizer_id, 0, "VHS_FILENAMES")

    graph["groups"].extend(
        [
            {
                "id": 53,
                "title": "10 IMAGES + 10 POSITIVE PROMPTS",
                "bounding": [-5210, 2920, 1110, 3440],
                "color": "#3f789e",
                "flags": {},
            },
            {
                "id": 54,
                "title": "BULK DROP + SEQUENTIAL QUEUE CONTROL",
                "bounding": [-4070, 2920, 700, 850],
                "color": "#5b4ca3",
                "flags": {},
            },
            {
                "id": 55,
                "title": "ZIP DOWNLOAD AFTER ALL 10",
                "bounding": [1360, 2920, 500, 300],
                "color": "#2f855a",
                "flags": {},
            },
        ]
    )

    by_id = {node["id"]: node for node in graph["nodes"]}
    by_id[323]["widgets_values"] = (
        "10-LOOP SEQUENTIAL QUEUE\n\n"
        "1. Drop a folder containing exactly 10 images onto the BULK DROP "
        "panel. A ZIP containing exactly 10 images is also accepted. Images "
        "are assigned by natural filename order (01, 02, ..., 10).\n"
        "2. Drop prompts.txt onto the same panel. Write each positive prompt "
        "as a block of one or more lines, and put an empty line between the "
        "10 blocks. Blocks are assigned in image order. Legacy one-line x10, "
        "JSON arrays, and blocks separated by --- are also accepted.\n"
        "3. Confirm READY, then press QUEUE 10 LOOPS (SEQUENTIAL) once.\n"
        "4. ComfyUI creates ten separate jobs. Only one loop occupies VRAM at "
        "a time.\n5. After job 10 succeeds, the browser downloads one ZIP with "
        "slot-01.mp4 through slot-10.mp4 plus manifest.json.\n\n"
        "Do not use the normal Queue button for this preset. A failed earlier "
        "job intentionally prevents an incomplete ZIP from downloading."
    )
    bundle = graph.setdefault("extra", {}).setdefault("runpod_bundle", {})
    bundle.update(
        {
            "preset": "seamless-loop-batch10-sequential",
            "profile": bundle.get("profile", "loop-all"),
            "queue_jobs": 10,
            "auto_download": "zip-after-final-job",
        }
    )
    return graph


def _auto_mosaic_node(node_id, position, order):
    return {
        "id": node_id,
        "type": "WanAutoMosaicVideo",
        "pos": list(position),
        "size": [410, 360],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            {"name": "images", "type": "IMAGE", "link": None},
            {
                "name": "model_name",
                "type": "COMBO",
                "widget": {"name": "model_name"},
                "link": None,
            },
            {
                "name": "coverage_preset",
                "type": "COMBO",
                "widget": {"name": "coverage_preset"},
                "link": None,
            },
            {
                "name": "confidence",
                "type": "FLOAT",
                "widget": {"name": "confidence"},
                "link": None,
            },
            {
                "name": "iou_threshold",
                "type": "FLOAT",
                "widget": {"name": "iou_threshold"},
                "link": None,
            },
            {
                "name": "block_size",
                "type": "INT",
                "widget": {"name": "block_size"},
                "link": None,
            },
            {
                "name": "max_gap_frames",
                "type": "INT",
                "widget": {"name": "max_gap_frames"},
                "link": None,
            },
            {
                "name": "target_classes",
                "type": "STRING",
                "widget": {"name": "target_classes"},
                "link": None,
            },
        ],
        "outputs": [
            {
                "name": "mosaicked_images",
                "type": "IMAGE",
                "links": [],
            }
        ],
        "properties": {"Node name for S&R": "WanAutoMosaicVideo"},
        "widgets_values": [
            "ntd11_anime_nsfw_segm_v5.pt",
            "JUST",
            0.30,
            0.50,
            0,
            3,
            "pussy,penis,testicles",
        ],
        "title": "AUTO MOSAIC JUST CONTOUR (CPU)",
    }


def patch_auto_mosaic(loop, batch10=False):
    """Insert CPU auto-mosaic after RIFE and before the only MP4 encode."""
    graph = copy.deepcopy(loop)
    by_id = {node["id"]: node for node in graph["nodes"]}
    combine = next(
        node for node in graph["nodes"] if node["type"] == "VHS_VideoCombine"
    )
    image_input = next(
        (index, item)
        for index, item in enumerate(combine["inputs"])
        if item["name"] == "images"
    )
    combine_slot, combine_images = image_input
    upstream_link = _top_level_link(graph, combine_images["link"])
    upstream = by_id[upstream_link[1]]
    if upstream["type"] != "RIFE VFI":
        raise ValueError(
            "auto mosaic must be inserted immediately after RIFE VFI; got "
            + str(upstream["type"])
        )

    mosaic_id = graph["last_node_id"] + 1
    mosaic = _auto_mosaic_node(
        mosaic_id,
        (1280, 2900),
        max(node.get("order", 0) for node in graph["nodes"]) + 1,
    )
    graph["nodes"].append(mosaic)
    graph["last_node_id"] = mosaic_id

    # Reuse the RIFE link as the mosaic input, then add a fresh mosaic->encode
    # link. This preserves both serialized endpoint link lists.
    upstream_link[3] = mosaic_id
    upstream_link[4] = 0
    mosaic["inputs"][0]["link"] = upstream_link[0]
    combine_images["link"] = None
    append_link(graph, mosaic_id, 0, combine["id"], combine_slot, "IMAGE")

    combine["pos"] = [1710, 3015]
    values = combine.get("widgets_values")
    if isinstance(values, dict) and not batch10:
        values["filename_prefix"] = (
            "Video/loops-mosaic/%date:yyyy-MM-dd%/%date:hhmmss%-loop-mosaic"
        )

    group_id = max(group["id"] for group in graph.get("groups", [])) + 1
    graph.setdefault("groups", []).append(
        {
            "id": group_id,
            "title": "POST-RIFE AUTO MOSAIC (CPU / LOOP-SAFE)",
            "bounding": [1240, 2800, 450, 500],
            "color": "#7a3f83",
            "flags": {},
        }
    )

    finalizer = next(
        (
            node
            for node in graph["nodes"]
            if node["type"] == "WanLoopBatchFinalize"
        ),
        None,
    )
    if finalizer:
        finalizer["pos"] = [2100, 3020]
        zip_group = next(
            (group for group in graph["groups"] if group["title"] == "ZIP DOWNLOAD AFTER ALL 10"),
            None,
        )
        if zip_group:
            zip_group["bounding"][0] = 2060

    note = by_id.get(323)
    if note:
        note["widgets_values"] = (
            "AUTO MOSAIC OUTPUT PRESET\n\n"
            "Mosaic is applied to completed frames after RIFE and before MP4 "
            "encoding. Anime NSFW Detection v5.0 produces a per-pixel instance "
            "segmentation mask on CPU, so WAN keeps exclusive GPU VRAM. JUST "
            "matches the AutoMosaic iPhone contour preset: segmentation only "
            "with a 4% mask dilation. Default targets are pussy, penis, and "
            "testicles; anus is deliberately excluded. block_size=0 automatically uses short "
            "side / 50 (minimum 10 px). max_gap_frames=3 interpolates only "
            "brief detector misses, including across the loop seam; it never "
            "unions neighboring masks into frames that already have a valid "
            "contour. WIDE and SAFE deliberately cover a larger ellipse.\n\n"
            + str(note.get("widgets_values") or "")
        )

    bundle = graph.setdefault("extra", {}).setdefault("runpod_bundle", {})
    bundle.update(
        {
            "preset": (
                "seamless-loop-batch10-sequential-auto-mosaic"
                if batch10
                else "seamless-loop-auto-mosaic"
            ),
            "profile": bundle.get("profile", "loop-all"),
            "postprocess": "Anime NSFW Detection v5 YOLO11-seg JUST contour mosaic after RIFE (CPU)",
            "requires_all_referenced_assets": True,
        }
    )
    return graph


def patch_lightning(graph):
    graph = replace_model_names(graph)

    # The source canvas contains two disconnected FP8 loaders (917/918), while
    # the actual generation graph is wired exclusively to the Q8 GGUF pair.
    # Keeping the dead loaders makes ComfyUI report two missing models and used
    # to make provisioning block startup on inaccessible CivitAI downloads.
    disconnected_fp8_loaders = {917, 918}
    linked_fp8_loaders = {
        endpoint
        for link in graph.get("links", [])
        for endpoint in (link[1], link[3])
        if endpoint in disconnected_fp8_loaders
    }
    if linked_fp8_loaders:
        raise ValueError(
            "source workflow connected an FP8 alternate; review Q8-only patch: "
            + ", ".join(map(str, sorted(linked_fp8_loaders)))
        )
    graph["nodes"] = [
        node
        for node in graph.get("nodes", [])
        if node.get("id") not in disconnected_fp8_loaders
    ]

    # Loader metadata in the source points at generic official Wan files, while
    # the actual widgets select the author's Enhanced Lightning checkpoints.
    # Remove that stale Manager metadata so ComfyUI cannot offer the wrong
    # 28-GB pair as a supposed fix for a missing model.
    for node in graph.get("nodes", []):
        properties = node.get("properties")
        if isinstance(properties, dict):
            properties.pop("models", None)

    high_loras = [
        lora(
            "SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors",
            1.0,
        ),
        lora(
            "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            3.0,
        ),
        lora(
            "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
            1.0,
        ),
    ]
    low_loras = [
        lora(
            "SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors",
            1.0,
        ),
        lora(
            "Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
            1.0,
        ),
        lora(
            "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
            1.0,
        ),
    ]
    for node in graph.get("nodes", []):
        if node.get("type") != "Power Lora Loader (rgthree)":
            continue
        title = str(node.get("title", "")).upper()
        configure_lora_node(
            node,
            copy.deepcopy(high_loras if "HIGH" in title else low_loras),
        )

    by_id = {node["id"]: node for node in graph.get("nodes", [])}
    note = by_id.get(301)
    if note and isinstance(note.get("widgets_values"), list):
        note["widgets_values"][0] = (
            "# RUNPOD BUNDLE\n\n"
            "Use `MODEL_PROFILE=lightning-longvideo`. This production bundle "
            "uses the connected Q8 High/Low route; the source ZIP's disconnected "
            "FP8 alternates are intentionally omitted. The selected Enhanced "
            "V2 checkpoint already includes Lightning. All optional LoRAs below "
            "are downloaded but intentionally OFF; do not stack another "
            "Lightning LoRA unless you deliberately want to retune motion.\n\n"
            + str(note["widgets_values"][0])
        )

    graph.setdefault("extra", {})["runpod_bundle"] = {
        "profile": "lightning-longvideo",
        "model_manifest": "config/wan22-models.json",
        "source": "WAN 2.2 Native Enhanced Lightning long-video",
        "checkpoint": "Enhanced FAST MOVE V2 Q8 High/Low (Lightning included)",
        "requires_all_referenced_assets": True,
    }
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

    smooth_source = json.loads(SMOOTH_SOURCE.read_text(encoding="utf-8-sig"))
    lightning_source = json.loads(
        LIGHTNING_SOURCE.read_text(encoding="utf-8-sig")
    )
    aio = patch_aio(smooth_source)
    loop = patch_loop(aio)
    loop_core = patch_loop_core(loop)
    batch10 = patch_loop_batch10(loop)
    batch10_core = patch_loop_batch10(loop_core)
    generated = {
        "aio": encode(aio),
        "loop": encode(loop),
        "loop_core": encode(loop_core),
        "batch10": encode(batch10),
        "batch10_core": encode(batch10_core),
        "loop_mosaic": encode(patch_auto_mosaic(loop)),
        "loop_mosaic_core": encode(patch_auto_mosaic(loop_core)),
        "batch10_mosaic": encode(patch_auto_mosaic(batch10, batch10=True)),
        "batch10_mosaic_core": encode(
            patch_auto_mosaic(batch10_core, batch10=True)
        ),
        "lightning": encode(patch_lightning(lightning_source)),
    }

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
