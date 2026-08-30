import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / "workflows" / "wan22_smooth_v6_aio_runpod.json",
    ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_runpod.json",
    ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_batch10_runpod.json",
)
MOSAIC_WORKFLOWS = (
    ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_auto_mosaic_runpod.json",
    ROOT
    / "workflows"
    / "wan22_smooth_v6_seamless_loop_batch10_auto_mosaic_runpod.json",
)
CORE_WORKFLOWS = (
    ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_core_runpod.json",
    ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_batch10_core_runpod.json",
    ROOT
    / "workflows"
    / "wan22_smooth_v6_seamless_loop_core_auto_mosaic_runpod.json",
    ROOT
    / "workflows"
    / "wan22_smooth_v6_seamless_loop_batch10_core_auto_mosaic_runpod.json",
)
LOOP_WORKFLOWS = (*WORKFLOWS[1:], *MOSAIC_WORKFLOWS, *CORE_WORKFLOWS)
I2V_MOSAIC = (
    ROOT / "workflows" / "wan22_smooth_v6_i2v_auto_mosaic_runpod.json"
)
LIGHTNING = (
    ROOT / "workflows" / "wan22_native_enhanced_lightning_longvideo_runpod.json"
)


class WorkflowWiringTests(unittest.TestCase):
    def load(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_sampler_subgraphs_are_flattened(self):
        aio_expected = {
            234: "KSamplerWithNAG (Advanced)",
            235: "KSamplerAdvanced",
            236: "KSamplerAdvanced",
            237: "KSamplerWithNAG (Advanced)",
            329: "KSamplerAdvanced",
            330: "KSamplerWithNAG (Advanced)",
        }
        expected_by_path = {
            WORKFLOWS[0]: aio_expected,
            WORKFLOWS[1]: {
                329: "KSamplerAdvanced",
                330: "KSamplerWithNAG (Advanced)",
            },
            WORKFLOWS[2]: {
                329: "KSamplerAdvanced",
                330: "KSamplerWithNAG (Advanced)",
            },
        }
        for path, expected in expected_by_path.items():
            with self.subTest(path=path.name):
                graph = self.load(path)
                by_id = {node["id"]: node for node in graph["nodes"]}
                self.assertEqual(
                    {node_id: by_id[node_id]["type"] for node_id in expected},
                    expected,
                )

    def test_nag_negative_fanout_is_preserved(self):
        node_ids_by_path = {
            WORKFLOWS[0]: (237, 330),
            WORKFLOWS[1]: (330,),
            WORKFLOWS[2]: (330,),
        }
        for path, node_ids in node_ids_by_path.items():
            with self.subTest(path=path.name):
                graph = self.load(path)
                by_id = {node["id"]: node for node in graph["nodes"]}
                links = {link[0]: link for link in graph["links"]}
                for node_id in node_ids:
                    node = by_id[node_id]
                    inputs = {item["name"]: item for item in node["inputs"]}
                    negative = links[inputs["negative"]["link"]]
                    nag_negative = links[inputs["nag_negative"]["link"]]
                    self.assertNotEqual(negative[0], nag_negative[0])
                    self.assertEqual(negative[1:3], nag_negative[1:3])

    def test_every_top_level_link_matches_its_declared_slots(self):
        for path in (
            *WORKFLOWS,
            *MOSAIC_WORKFLOWS,
            *CORE_WORKFLOWS,
            I2V_MOSAIC,
            LIGHTNING,
        ):
            with self.subTest(path=path.name):
                graph = self.load(path)
                links = {link[0]: link for link in graph["links"]}
                for node in graph["nodes"]:
                    for slot, item in enumerate(node.get("inputs", [])):
                        link_id = item.get("link")
                        if link_id is None:
                            continue
                        self.assertIn(link_id, links)
                        self.assertEqual(links[link_id][3:5], [node["id"], slot])
                    for slot, item in enumerate(node.get("outputs", [])):
                        for link_id in item.get("links") or []:
                            self.assertIn(link_id, links)
                            self.assertEqual(links[link_id][1:3], [node["id"], slot])

    def test_vaedecode_receives_sampler_latent_not_model(self):
        for path in WORKFLOWS:
            with self.subTest(path=path.name):
                graph = self.load(path)
                by_id = {node["id"]: node for node in graph["nodes"]}
                links = {link[0]: link for link in graph["links"]}
                decode = by_id[384]
                cleanup = by_id[links[decode["inputs"][0]["link"]][1]]
                sampler = by_id[links[cleanup["inputs"][0]["link"]][1]]
                self.assertEqual(sampler["type"], "KSamplerWithNAG (Advanced)")
                self.assertEqual(sampler["outputs"][0]["type"], "LATENT")

    def test_loop_workflow_has_requested_lora_configuration(self):
        graph = self.load(WORKFLOWS[1])
        entries = [
            item
            for node in graph["nodes"]
            if node["type"] == "Power Lora Loader (rgthree)"
            for item in node["widgets_values"]
            if isinstance(item, dict) and item.get("lora")
        ]
        self.assertEqual(len(entries), 22)
        self.assertEqual(
            {item["lora"] for item in entries},
            {
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
                "NSFW-22-H-e8.safetensors",
                "NSFW-22-L-e8.safetensors",
                "SmoothXXXAnimation_High.safetensors",
                "SmoothXXXAnimation_Low.safetensors",
                "Cumshot_Aesthetics_High.safetensors",
                "Cumshot_Aesthetics_Low.safetensors",
                "I2V_joi_trend_high.safetensors",
                "I2V_joi_trend_low.safetensors",
                "Wan22_ThroatV3_High.safetensors",
                "Wan22_ThroatV3_Low.safetensors",
                "cheek_bulge_fellatio_high_wan-2-2_i2v_A14B.safetensors",
                "glans_licking_high_wan-2-2_i2v_A14B.safetensors",
                "head_back_high_wan-2-2_i2v_A14B.safetensors",
                "paizuri_unaligned_breasts_high_wan-2-2_i2v_A14B.safetensors",
                "washizukami_high_wan-2-2_i2v_A14B.safetensors",
                "cheek_bulge_fellatio_wanvideo_i2v.safetensors",
                "glans_licking_wanvideo_i2v_epoch5.safetensors",
                "head_back_wanvideo_i2v_epoch5.safetensors",
                "paizuri_unaligned_breasts_wanvideo_i2v_epoch5.safetensors",
                "washizukami_wanvideo_i2v.safetensors",
            },
        )
        active = {item["lora"] for item in entries if item["on"] is True}
        self.assertEqual(
            active,
            {
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
                "NSFW-22-H-e8.safetensors",
                "NSFW-22-L-e8.safetensors",
                "SmoothXXXAnimation_High.safetensors",
                "SmoothXXXAnimation_Low.safetensors",
            },
        )
        nsfw = [item for item in entries if item["lora"].startswith("NSFW-22-")]
        self.assertEqual(
            {item["lora"]: item["strength"] for item in nsfw},
            {
                "NSFW-22-H-e8.safetensors": 2.75,
                "NSFW-22-L-e8.safetensors": 1.65,
            },
        )
        smooth_xxx = [
            item for item in entries if item["lora"].startswith("SmoothXXXAnimation_")
        ]
        self.assertEqual(
            {item["lora"]: item["strength"] for item in smooth_xxx},
            {
                "SmoothXXXAnimation_High.safetensors": 1.5,
                "SmoothXXXAnimation_Low.safetensors": 1.0,
            },
        )
        self.assertTrue(all(item["on"] is True for item in smooth_xxx))
        cumshot = [
            item for item in entries if item["lora"].startswith("Cumshot_Aesthetics_")
        ]
        self.assertEqual(
            {item["lora"]: item["strength"] for item in cumshot},
            {
                "Cumshot_Aesthetics_High.safetensors": 1.0,
                "Cumshot_Aesthetics_Low.safetensors": 1.0,
            },
        )
        self.assertTrue(all(item["on"] is False for item in cumshot))
        joi = [
            item for item in entries if item["lora"].startswith("I2V_joi_trend_")
        ]
        self.assertEqual(
            {item["lora"]: item["strength"] for item in joi},
            {
                "I2V_joi_trend_high.safetensors": 1.0,
                "I2V_joi_trend_low.safetensors": 1.0,
            },
        )
        self.assertTrue(all(item["on"] is False for item in joi))
        throat = [
            item for item in entries if item["lora"].startswith("Wan22_ThroatV3_")
        ]
        self.assertEqual(
            {item["lora"]: item["strength"] for item in throat},
            {
                "Wan22_ThroatV3_High.safetensors": 1.0,
                "Wan22_ThroatV3_Low.safetensors": 1.0,
            },
        )
        self.assertTrue(all(item["on"] is False for item in throat))
        iroiro = [
            item
            for item in entries
            if item["lora"].endswith("_high_wan-2-2_i2v_A14B.safetensors")
        ]
        self.assertEqual(len(iroiro), 5)
        self.assertTrue(all(item["strength"] == 1.0 for item in iroiro))
        self.assertTrue(all(item["on"] is False for item in iroiro))
        iroiro_low = [
            item
            for item in entries
            if item["lora"]
            in {
                "cheek_bulge_fellatio_wanvideo_i2v.safetensors",
                "glans_licking_wanvideo_i2v_epoch5.safetensors",
                "head_back_wanvideo_i2v_epoch5.safetensors",
                "paizuri_unaligned_breasts_wanvideo_i2v_epoch5.safetensors",
                "washizukami_wanvideo_i2v.safetensors",
            }
        ]
        self.assertEqual(len(iroiro_low), 5)
        self.assertTrue(all(item["strength"] == 1.0 for item in iroiro_low))
        self.assertTrue(all(item["on"] is False for item in iroiro_low))

    def test_loop_workflow_uses_requested_resolution(self):
        for path in (*WORKFLOWS[1:], *CORE_WORKFLOWS):
            with self.subTest(path=path.name):
                graph = self.load(path)
                node = next(node for node in graph["nodes"] if node["id"] == 328)
                self.assertEqual(node["properties"]["valueX"], 528)
                self.assertEqual(node["properties"]["valueY"], 704)
                self.assertEqual(node["widgets_values"], [528, 528, 704, 704, 0, 0])

    def test_batch10_inherits_optional_cumshot_lora_pair(self):
        expected = {
            "Cumshot_Aesthetics_High.safetensors": (1.0, False),
            "Cumshot_Aesthetics_Low.safetensors": (1.0, False),
        }
        for path in WORKFLOWS[1:]:
            with self.subTest(path=path.name):
                graph = self.load(path)
                entries = {
                    item["lora"]: (item["strength"], item["on"])
                    for node in graph["nodes"]
                    if node["type"] == "Power Lora Loader (rgthree)"
                    for item in node["widgets_values"]
                    if isinstance(item, dict)
                    and item.get("lora", "").startswith("Cumshot_Aesthetics_")
                }
                self.assertEqual(entries, expected)

    def test_all_loop_variants_inherit_optional_joi_lora_pair(self):
        expected = {
            "I2V_joi_trend_high.safetensors": (1.0, False),
            "I2V_joi_trend_low.safetensors": (1.0, False),
        }
        for path in WORKFLOWS[1:]:
            with self.subTest(path=path.name):
                graph = self.load(path)
                entries = {
                    item["lora"]: (item["strength"], item["on"])
                    for node in graph["nodes"]
                    if node["type"] == "Power Lora Loader (rgthree)"
                    for item in node["widgets_values"]
                    if isinstance(item, dict)
                    and item.get("lora", "").startswith("I2V_joi_trend_")
                }
                self.assertEqual(entries, expected)

    def test_all_non_core_loop_variants_inherit_optional_throat_lora_pair(self):
        for path in (*WORKFLOWS[1:], *MOSAIC_WORKFLOWS):
            with self.subTest(path=path.name):
                graph = self.load(path)
                by_id = {node["id"]: node for node in graph["nodes"]}
                high = {
                    item["lora"]: (item["strength"], item["on"])
                    for item in by_id[325]["widgets_values"]
                    if isinstance(item, dict)
                    and item.get("lora", "").startswith("Wan22_ThroatV3_")
                }
                low = {
                    item["lora"]: (item["strength"], item["on"])
                    for item in by_id[324]["widgets_values"]
                    if isinstance(item, dict)
                    and item.get("lora", "").startswith("Wan22_ThroatV3_")
                }
                self.assertEqual(
                    high, {"Wan22_ThroatV3_High.safetensors": (1.0, False)}
                )
                self.assertEqual(
                    low, {"Wan22_ThroatV3_Low.safetensors": (1.0, False)}
                )

    def test_batch10_inherits_optional_iroiro_high_low_pairs(self):
        expected_high = {
            "cheek_bulge_fellatio_high_wan-2-2_i2v_A14B.safetensors": (1.0, False),
            "glans_licking_high_wan-2-2_i2v_A14B.safetensors": (1.0, False),
            "head_back_high_wan-2-2_i2v_A14B.safetensors": (1.0, False),
            "paizuri_unaligned_breasts_high_wan-2-2_i2v_A14B.safetensors": (1.0, False),
            "washizukami_high_wan-2-2_i2v_A14B.safetensors": (1.0, False),
        }
        expected_low = {
            "cheek_bulge_fellatio_wanvideo_i2v.safetensors": (1.0, False),
            "glans_licking_wanvideo_i2v_epoch5.safetensors": (1.0, False),
            "head_back_wanvideo_i2v_epoch5.safetensors": (1.0, False),
            "paizuri_unaligned_breasts_wanvideo_i2v_epoch5.safetensors": (1.0, False),
            "washizukami_wanvideo_i2v.safetensors": (1.0, False),
        }
        for path in WORKFLOWS[1:]:
            with self.subTest(path=path.name):
                graph = self.load(path)
                high_loader = next(node for node in graph["nodes"] if node["id"] == 325)
                low_loader = next(node for node in graph["nodes"] if node["id"] == 324)
                high_entries = {
                    item["lora"]: (item["strength"], item["on"])
                    for item in high_loader["widgets_values"]
                    if isinstance(item, dict)
                    and item.get("lora", "").endswith(
                        "_high_wan-2-2_i2v_A14B.safetensors"
                    )
                }
                low_entries = {
                    item["lora"]: (item["strength"], item["on"])
                    for item in low_loader["widgets_values"]
                    if isinstance(item, dict) and item.get("lora") in expected_low
                }
                self.assertEqual(high_entries, expected_high)
                self.assertEqual(low_entries, expected_low)

    def test_loop_workflow_has_no_unused_model_loaders(self):
        for path in WORKFLOWS[1:]:
            with self.subTest(path=path.name):
                graph = self.load(path)
                types = {node["type"] for node in graph["nodes"]}
                self.assertNotIn("UnetLoaderGGUF", types)
                self.assertNotIn("MMAudioModelLoader", types)
                self.assertNotIn("MMAudioFeatureUtilsLoader", types)

    def test_batch10_is_ten_separate_queue_slots(self):
        graph = self.load(WORKFLOWS[2])
        nodes = graph["nodes"]
        slots = [node for node in nodes if node["type"] == "WanLoopQueueSlot"]
        selectors = [
            node for node in nodes if node["type"] == "WanLoopQueueSelector"
        ]
        finalizers = [
            node for node in nodes if node["type"] == "WanLoopBatchFinalize"
        ]
        self.assertEqual(len(slots), 10)
        self.assertEqual(len(selectors), 1)
        self.assertEqual(len(finalizers), 1)
        self.assertEqual(selectors[0]["widgets_values"][:2], [1, "increment"])
        self.assertEqual(finalizers[0]["widgets_values"], [10])
        self.assertEqual(
            graph["extra"]["runpod_bundle"]["preset"],
            "seamless-loop-batch10-sequential",
        )
        self.assertEqual(graph["extra"]["runpod_bundle"]["queue_jobs"], 10)

    def test_every_batch10_variant_exposes_bulk_drop_selector(self):
        batch_paths = (
            WORKFLOWS[2],
            CORE_WORKFLOWS[1],
            MOSAIC_WORKFLOWS[1],
            CORE_WORKFLOWS[3],
        )
        for path in batch_paths:
            with self.subTest(path=path.name):
                graph = self.load(path)
                selector = next(
                    node
                    for node in graph["nodes"]
                    if node["type"] == "WanLoopQueueSelector"
                )
                self.assertEqual(selector["size"], [600, 720])
                self.assertIn("BULK DROP", selector["title"])
                note = next(node for node in graph["nodes"] if node["id"] == 323)
                self.assertIn("exactly 10 images", note["widgets_values"])
                self.assertIn("prompts.txt", note["widgets_values"])
                self.assertIn("empty line between", note["widgets_values"])

    def test_batch10_reuses_one_selected_image_at_both_loop_ends(self):
        graph = self.load(WORKFLOWS[2])
        by_id = {node["id"]: node for node in graph["nodes"]}
        links = {link[0]: link for link in graph["links"]}
        self.assertNotIn(333, by_id)
        self.assertNotIn(338, by_id)
        self.assertNotIn(342, by_id)

        selector = next(
            node for node in graph["nodes"] if node["type"] == "WanLoopQueueSelector"
        )
        selector_id = selector["id"]
        image_targets = {
            (link[3], link[4])
            for link in graph["links"]
            if link[1:3] == [selector_id, 0]
        }
        self.assertEqual(image_targets, {(352, 1), (350, 1), (343, 5), (343, 6)})

        prompt_link = links[by_id[305]["inputs"][0]["link"]]
        prefix_link = links[by_id[332]["inputs"][6]["link"]]
        self.assertEqual(prompt_link[1:3], [selector_id, 1])
        self.assertEqual(prefix_link[1:3], [selector_id, 2])
        self.assertEqual(by_id[343]["widgets_values"][3], 1)

    def test_batch10_finalizer_is_downstream_of_saved_video(self):
        graph = self.load(WORKFLOWS[2])
        by_id = {node["id"]: node for node in graph["nodes"]}
        links = {link[0]: link for link in graph["links"]}
        finalizer = next(
            node for node in graph["nodes"] if node["type"] == "WanLoopBatchFinalize"
        )
        selector = next(
            node for node in graph["nodes"] if node["type"] == "WanLoopQueueSelector"
        )
        filenames_link = links[finalizer["inputs"][0]["link"]]
        context_link = links[finalizer["inputs"][1]["link"]]
        self.assertEqual(by_id[filenames_link[1]]["type"], "VHS_VideoCombine")
        self.assertEqual(filenames_link[2], 0)
        self.assertEqual(context_link[1:3], [selector["id"], 3])

    def test_auto_mosaic_is_a_separate_post_rife_pre_encode_workflow(self):
        for path in WORKFLOWS:
            graph = self.load(path)
            self.assertNotIn(
                "WanAutoMosaicVideo", {node["type"] for node in graph["nodes"]}
            )

        for path in MOSAIC_WORKFLOWS:
            with self.subTest(path=path.name):
                graph = self.load(path)
                by_id = {node["id"]: node for node in graph["nodes"]}
                links = {link[0]: link for link in graph["links"]}
                mosaics = [
                    node
                    for node in graph["nodes"]
                    if node["type"] == "WanAutoMosaicVideo"
                ]
                self.assertEqual(len(mosaics), 1)
                mosaic = mosaics[0]
                input_link = links[mosaic["inputs"][0]["link"]]
                self.assertEqual(by_id[input_link[1]]["type"], "RIFE VFI")

                combine = next(
                    node
                    for node in graph["nodes"]
                    if node["type"] == "VHS_VideoCombine"
                )
                image_input = next(
                    item for item in combine["inputs"] if item["name"] == "images"
                )
                output_link = links[image_input["link"]]
                self.assertEqual(output_link[1:3], [mosaic["id"], 0])
                self.assertEqual(
                    mosaic["widgets_values"],
                    [
                        "ntd11_anime_nsfw_segm_v5.pt",
                        "JUST",
                        0.3,
                        0.5,
                        0,
                        3,
                        "pussy,penis,testicles",
                    ],
                )
                self.assertEqual(
                    graph["extra"]["runpod_bundle"]["profile"], "loop-all"
                )
                note = next(node for node in graph["nodes"] if node["id"] == 323)
                self.assertIn("anus is deliberately excluded", note["widgets_values"])

    def test_normal_i2v_mosaic_is_single_image_non_loop_and_post_rife(self):
        graph = self.load(I2V_MOSAIC)
        by_id = {node["id"]: node for node in graph["nodes"]}
        links = {link[0]: link for link in graph["links"]}
        types = [node["type"] for node in graph["nodes"]]

        self.assertEqual(types.count("LoadImage"), 1)
        self.assertEqual(types.count("WanImageToVideo"), 1)
        self.assertNotIn("WanFirstLastFrameToVideo", types)
        self.assertNotIn("MMAudioModelLoader", types)
        self.assertNotIn("MMAudioFeatureUtilsLoader", types)
        self.assertNotIn("UnetLoaderGGUF", types)

        conditioning = next(
            node for node in graph["nodes"] if node["type"] == "WanImageToVideo"
        )
        start_image = next(
            item for item in conditioning["inputs"] if item["name"] == "start_image"
        )
        resized = by_id[links[start_image["link"]][1]]
        self.assertEqual(resized["type"], "ImageResizeKJv2")
        source = by_id[links[resized["inputs"][0]["link"]][1]]
        self.assertEqual(source["type"], "LoadImage")

        mosaic = next(
            node for node in graph["nodes"] if node["type"] == "WanAutoMosaicVideo"
        )
        mosaic_input = by_id[links[mosaic["inputs"][0]["link"]][1]]
        self.assertEqual(mosaic_input["type"], "RIFE VFI")
        combine = next(
            node for node in graph["nodes"] if node["type"] == "VHS_VideoCombine"
        )
        combine_image = next(
            item for item in combine["inputs"] if item["name"] == "images"
        )
        self.assertEqual(links[combine_image["link"]][1:3], [mosaic["id"], 0])
        self.assertIn("i2v-mosaic", combine["widgets_values"]["filename_prefix"])

        self.assertEqual(by_id[208]["properties"]["valueX"], 528)
        self.assertEqual(by_id[208]["properties"]["valueY"], 704)
        self.assertEqual(by_id[178]["widgets_values"], [5])
        self.assertEqual(
            graph["extra"]["runpod_bundle"]["preset"], "i2v-auto-mosaic"
        )

        def rect(item, key):
            x, y, width, height = map(float, item[key])
            return (x, y, x + width, y + height)

        def node_rect(node):
            x, y = map(float, node["pos"])
            width, height = map(float, node.get("size", [220, 80])[:2])
            return (x, y, x + width, y + height)

        def overlaps(first, second):
            return (
                min(first[2], second[2]) - max(first[0], second[0]) > 0.01
                and min(first[3], second[3]) - max(first[1], second[1]) > 0.01
            )

        for index, first in enumerate(graph["nodes"]):
            for second in graph["nodes"][index + 1 :]:
                self.assertFalse(
                    overlaps(node_rect(first), node_rect(second)),
                    f"normal I2V nodes {first['id']} and {second['id']} overlap",
                )
        child_groups = [group for group in graph["groups"] if group["id"] != 34]
        for index, first in enumerate(child_groups):
            for second in child_groups[index + 1 :]:
                self.assertFalse(
                    overlaps(rect(first, "bounding"), rect(second, "bounding")),
                    f"normal I2V groups {first['id']} and {second['id']} overlap",
                )

    def test_normal_i2v_mosaic_exposes_throat_v3_high_low_pair(self):
        graph = self.load(I2V_MOSAIC)
        entries = {
            item["lora"]: (item["strength"], item["on"])
            for node in graph["nodes"]
            if node["type"] == "Power Lora Loader (rgthree)"
            for item in node["widgets_values"]
            if isinstance(item, dict)
            and item.get("lora", "").startswith("Wan22_ThroatV3_")
        }
        self.assertEqual(
            entries,
            {
                "Wan22_ThroatV3_High.safetensors": (1.0, False),
                "Wan22_ThroatV3_Low.safetensors": (1.0, False),
            },
        )

    def test_loop_variants_use_nmkd_model_upscale_at_net_two_x(self):
        for path in LOOP_WORKFLOWS:
            with self.subTest(path=path.name):
                graph = self.load(path)
                by_id = {node["id"]: node for node in graph["nodes"]}
                links = {link[0]: link for link in graph["links"]}
                loaders = [
                    node
                    for node in graph["nodes"]
                    if node["type"] == "UpscaleModelLoader"
                ]
                upscalers = [
                    node
                    for node in graph["nodes"]
                    if node["type"] == "ImageUpscaleWithModel"
                ]
                self.assertEqual(len(loaders), 1)
                self.assertEqual(len(upscalers), 1)
                loader = loaders[0]
                upscaler = upscalers[0]
                self.assertEqual(
                    loader["widgets_values"], "4x_NMKD-Siax_200k.pth"
                )

                upscale_inputs = {item["name"]: item for item in upscaler["inputs"]}
                model_link = links[upscale_inputs["upscale_model"]["link"]]
                image_link = links[upscale_inputs["image"]["link"]]
                self.assertEqual(model_link[1:3], [loader["id"], 0])
                self.assertEqual(by_id[image_link[1]]["type"], "VAEDecode")

                downscale = by_id[320]
                self.assertEqual(downscale["widgets_values"], ["nearest-exact", 0.5])
                downscale_link = links[downscale["inputs"][0]["link"]]
                self.assertEqual(downscale_link[1:3], [upscaler["id"], 0])
                self.assertFalse(
                    any(
                        node["type"] == "ImageScaleBy"
                        and node.get("widgets_values") == ["lanczos", 2.0]
                        for node in graph["nodes"]
                    )
                )

    def test_loop_canvases_are_compact_aligned_and_non_overlapping(self):
        def node_rect(node):
            x, y = map(float, node["pos"])
            width, height = map(float, node.get("size", [220, 80])[:2])
            return (x, y, x + width, y + height)

        def group_rect(group):
            x, y, width, height = map(float, group["bounding"])
            return (x, y, x + width, y + height)

        def overlaps(first, second):
            return (
                min(first[2], second[2]) - max(first[0], second[0]) > 0.01
                and min(first[3], second[3]) - max(first[1], second[1]) > 0.01
            )

        def contains(outer, inner):
            return (
                inner[0] >= outer[0] - 0.01
                and inner[1] >= outer[1] - 0.01
                and inner[2] <= outer[2] + 0.01
                and inner[3] <= outer[3] + 0.01
            )

        removed_source_decorations = {
            105,
            106,
            263,
            264,
            306,
            307,
            309,
            310,
            386,
            387,
            388,
            397,
            398,
            403,
        }
        for path in LOOP_WORKFLOWS:
            with self.subTest(path=path.name):
                graph = self.load(path)
                nodes = graph["nodes"]
                groups = graph["groups"]
                self.assertTrue(
                    removed_source_decorations.isdisjoint(
                        {node["id"] for node in nodes}
                    )
                )

                for index, first in enumerate(nodes):
                    for second in nodes[index + 1 :]:
                        self.assertFalse(
                            overlaps(node_rect(first), node_rect(second)),
                            f"nodes {first['id']} and {second['id']} overlap",
                        )

                parent = next(group for group in groups if group["id"] == 36)
                parent_bounds = group_rect(parent)
                children = [group for group in groups if group["id"] != 36]
                for node in nodes:
                    self.assertTrue(contains(parent_bounds, node_rect(node)))
                    memberships = [
                        group["id"]
                        for group in children
                        if contains(group_rect(group), node_rect(node))
                    ]
                    self.assertEqual(
                        memberships,
                        memberships[:1],
                        f"node {node['id']} is inside multiple groups",
                    )
                    self.assertEqual(
                        len(memberships),
                        1,
                        f"node {node['id']} is outside every lane",
                    )

                for index, first in enumerate(children):
                    for second in children[index + 1 :]:
                        self.assertFalse(
                            overlaps(group_rect(first), group_rect(second)),
                            f"groups {first['id']} and {second['id']} overlap",
                        )

    def test_core_variants_only_reference_enabled_loras(self):
        for path in CORE_WORKFLOWS:
            with self.subTest(path=path.name):
                graph = self.load(path)
                entries = [
                    item
                    for node in graph["nodes"]
                    if node["type"] == "Power Lora Loader (rgthree)"
                    for item in node["widgets_values"]
                    if isinstance(item, dict) and item.get("lora")
                ]
                self.assertEqual(len(entries), 6)
                self.assertTrue(all(item["on"] is True for item in entries))
                self.assertEqual(
                    graph["extra"]["runpod_bundle"]["profile"], "loop-core"
                )

    def test_batch10_mosaic_still_finalizes_the_encoded_files(self):
        graph = self.load(MOSAIC_WORKFLOWS[1])
        by_id = {node["id"]: node for node in graph["nodes"]}
        links = {link[0]: link for link in graph["links"]}
        finalizer = next(
            node for node in graph["nodes"] if node["type"] == "WanLoopBatchFinalize"
        )
        filenames = links[finalizer["inputs"][0]["link"]]
        self.assertEqual(by_id[filenames[1]]["type"], "VHS_VideoCombine")

    def test_lightning_models_are_normalized_to_manifest_names(self):
        graph = self.load(LIGHTNING)
        by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertNotIn(917, by_id)
        self.assertNotIn(918, by_id)
        self.assertEqual(
            by_id[919]["widgets_values"][0],
            "wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2Q8H.gguf",
        )
        self.assertEqual(
            by_id[920]["widgets_values"][0],
            "wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2Q8L.gguf",
        )

    def test_lightning_optional_loras_are_present_but_disabled(self):
        graph = self.load(LIGHTNING)
        nodes = [
            node
            for node in graph["nodes"]
            if node["type"] == "Power Lora Loader (rgthree)"
        ]
        self.assertEqual(len(nodes), 8)
        for node in nodes:
            entries = [
                item
                for item in node["widgets_values"]
                if isinstance(item, dict) and item.get("lora")
            ]
            self.assertEqual(len(entries), 3)
            self.assertTrue(all(item["on"] is False for item in entries))


if __name__ == "__main__":
    unittest.main()
