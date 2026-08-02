import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / "workflows" / "wan22_smooth_v6_aio_runpod.json",
    ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_runpod.json",
    ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_batch10_runpod.json",
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
        for path in (*WORKFLOWS, LIGHTNING):
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
        self.assertEqual(len(entries), 6)
        self.assertEqual(
            {item["lora"] for item in entries},
            {
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
                "NSFW-22-H-e8.safetensors",
                "NSFW-22-L-e8.safetensors",
                "SmoothXXXAnimation_High.safetensors",
                "SmoothXXXAnimation_Low.safetensors",
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

    def test_loop_workflow_uses_requested_resolution(self):
        for path in WORKFLOWS[1:]:
            with self.subTest(path=path.name):
                graph = self.load(path)
                node = next(node for node in graph["nodes"] if node["id"] == 328)
                self.assertEqual(node["properties"]["valueX"], 528)
                self.assertEqual(node["properties"]["valueY"], 704)
                self.assertEqual(node["widgets_values"], [528, 528, 704, 704, 0, 0])

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
