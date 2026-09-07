"""Contracts for the two shipped canvases, including frontend widget ordering."""
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SINGLE = ROOT / "workflows/wan22_loop_single_runpod.json"
BATCH = ROOT / "workflows/wan22_loop_batch10_runpod.json"
WORKFLOWS = (SINGLE, BATCH)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class WorkflowWiringTests(unittest.TestCase):
    def test_only_two_canvases_are_shipped(self):
        self.assertEqual(set((ROOT / "workflows").glob("*.json")), set(WORKFLOWS))

    def test_every_link_has_matching_input_and_output_slots(self):
        for path in WORKFLOWS:
            with self.subTest(path=path.name):
                graph = load(path)
                nodes = {n["id"]: n for n in graph["nodes"]}
                links = {link[0]: link for link in graph["links"]}
                self.assertEqual(len(nodes), len(graph["nodes"]))
                self.assertEqual(len(links), len(graph["links"]))
                for link_id, src, src_slot, dst, dst_slot, _type in links.values():
                    self.assertIn(link_id, nodes[src]["outputs"][src_slot]["links"])
                    self.assertEqual(nodes[dst]["inputs"][dst_slot]["link"], link_id)
                for node in nodes.values():
                    for slot, item in enumerate(node.get("inputs", [])):
                        if item.get("link") is not None:
                            self.assertEqual(links[item["link"]][3:5], [node["id"], slot])
                    for slot, item in enumerate(node.get("outputs", [])):
                        for link_id in item.get("links") or []:
                            self.assertEqual(links[link_id][1:3], [node["id"], slot])

    def test_enhanced_models_and_two_plus_two_schedule(self):
        for path in WORKFLOWS:
            nodes = {n["id"]: n for n in load(path)["nodes"]}
            for node_id, suffix in ((315, "High"), (316, "Low")):
                self.assertEqual(nodes[node_id]["type"], "UnetLoaderGGUF")
                self.assertEqual(nodes[node_id]["widgets_values"], [
                    f"wan22EnhancedNSFWSVICamera_nsfwV2Q8{suffix}.gguf"])
            for node_id in (329, 330):
                self.assertEqual(nodes[node_id]["type"], "KSamplerAdvanced")
                self.assertEqual(nodes[node_id]["widgets_values"][3:7], [4, 1, "euler", "simple"])
                self.assertFalse(any(i["name"].startswith("nag_") for i in nodes[node_id]["inputs"]))
            self.assertEqual(nodes[329]["widgets_values"][7:], [0, 2, "enable"])
            self.assertEqual(nodes[330]["widgets_values"][7:], [2, 4, "disable"])
            self.assertEqual(nodes[329]["widgets_values"][0], "enable")
            self.assertEqual(nodes[330]["widgets_values"][0], "disable")

    def test_optional_loras_are_not_removed_or_enabled(self):
        manifest = json.loads((ROOT / "config/wan22-models.json").read_text(encoding="utf-8"))
        groups = set(manifest["profiles"]["loop-all"]["include_groups"])
        expected = {pathlib.PurePosixPath(m["path"]).name for m in manifest["models"]
                    if m["group"] in groups and m["path"].startswith("models/loras/")}
        for path in WORKFLOWS:
            entries = [(n["id"], w) for n in load(path)["nodes"]
                if n["type"] == "Power Lora Loader (rgthree)"
                for w in n["widgets_values"] if isinstance(w, dict) and w.get("lora")]
            self.assertEqual({w["lora"] for _, w in entries}, expected)
            self.assertEqual(len(entries), 21)
            self.assertTrue(all(w["on"] is False and w["strength"] == 1.0 for _, w in entries))
            self.assertEqual([node_id for node_id, w in entries if w["lora"] == "wind.safetensors"], [324])

    def test_native_resolution_and_duration_are_not_silently_reduced(self):
        for path in WORKFLOWS:
            nodes = {n["id"]: n for n in load(path)["nodes"]}
            self.assertEqual(nodes[328]["widgets_values"], [720, 720, 960, 960, 0, 0])
            self.assertEqual(nodes[328]["properties"]["valueX"], 720)
            self.assertEqual(nodes[328]["properties"]["valueY"], 960)
            self.assertEqual(nodes[335]["widgets_values"], [5])
            self.assertEqual(nodes[321]["widgets_values"], ["a*16+1"])

    def test_upscale_combo_restores_full_filename_not_first_character(self):
        for path in WORKFLOWS:
            node = next(n for n in load(path)["nodes"] if n["type"] == "UpscaleModelLoader")
            values = node["widgets_values"]
            # LiteGraph restores positional widgets by index. A string here
            # silently selected '4' instead of the original 4x filename.
            self.assertIsInstance(values, list)
            self.assertEqual(len(values), 1)
            self.assertEqual(values[0], "2xNomosUni_span_multijpg.safetensors")
            self.assertEqual(node["inputs"][0]["widget"]["name"], "model_name")

    def test_postprocessing_order_and_vram_bounded_defaults(self):
        for path in WORKFLOWS:
            graph = load(path)
            nodes = {n["id"]: n for n in graph["nodes"]}
            links = {link[0]: link for link in graph["links"]}
            def upstream(node, name):
                item = next(i for i in node["inputs"] if i["name"] == name)
                return nodes[links[item["link"]][1]]
            upscale = next(n for n in nodes.values() if n["type"] == "WanLoopModelUpscale")
            self.assertEqual(upscale["widgets_values"], [2.0])
            self.assertEqual(upstream(upscale, "image")["type"], "RIFE VFI")
            self.assertEqual(upstream(upscale, "upscale_model")["type"], "UpscaleModelLoader")
            self.assertFalse(any(n["type"] == "ImageScaleBy" for n in nodes.values()))
            latent_clean = upstream(nodes[384], "samples")
            self.assertEqual(upstream(latent_clean, "anything")["id"], 330)
            rife = next(n for n in nodes.values() if n["type"] == "RIFE VFI")
            self.assertEqual(rife["widgets_values"], ["rife49.pth", 10, 2, True, False, 1, "float16", False, 1])
            self.assertEqual(upstream(upstream(rife, "frames"), "anything")["type"], "VAEDecode")
            mosaic = next(n for n in nodes.values() if n["type"] == "WanAutoMosaicVideo")
            self.assertEqual(upstream(mosaic, "images")["id"], upscale["id"])
            self.assertEqual(mosaic["widgets_values"], [
                "ntd11_anime_nsfw_segm_v5.pt", "JUST", 0.3, 0.5, 0, 3, "pussy,penis,testicles", "auto"])
            combine = [n for n in nodes.values() if n["type"] == "VHS_VideoCombine"]
            self.assertEqual(len(combine), 1)
            self.assertEqual(upstream(combine[0], "images")["id"], mosaic["id"])
            self.assertEqual(combine[0]["widgets_values"]["frame_rate"], 32)
            self.assertFalse(combine[0]["widgets_values"]["pingpong"])

    def test_single_and_batch_share_identical_generation_settings(self):
        a, b = [{n["id"]: n for n in load(p)["nodes"]} for p in WORKFLOWS]
        for node_id in (297, 298, 315, 316, 324, 325, 328, 329, 330, 335, 341, 399, 404, 405):
            self.assertEqual(a[node_id]["widgets_values"], b[node_id]["widgets_values"])

    def test_batch_uses_ten_slots_but_only_one_inference_chain(self):
        graph = load(BATCH)
        nodes = graph["nodes"]
        self.assertEqual(sum(n["type"] == "WanLoopQueueSlot" for n in nodes), 10)
        self.assertEqual(sum(n["type"] == "KSamplerAdvanced" for n in nodes), 2)
        selectors = [n for n in nodes if n["type"] == "WanLoopQueueSelector"]
        self.assertEqual(len(selectors), 1)
        selector = selectors[0]
        self.assertEqual(sum(i["name"].startswith("slot_") for i in selector["inputs"]), 10)
        links = {link[0]: link for link in graph["links"]}
        conditioning = next(n for n in nodes if n["type"] == "WanFirstLastFrameToVideo")
        for name in ("start_image", "end_image"):
            item = next(i for i in conditioning["inputs"] if i["name"] == name)
            self.assertEqual(links[item["link"]][1:3], [selector["id"], 0])
        finalizer = next(n for n in nodes if n["type"] == "WanLoopBatchFinalize")
        files = next(i for i in finalizer["inputs"] if i["name"] == "filenames")
        source = next(n for n in nodes if n["id"] == links[files["link"]][1])
        self.assertEqual(source["type"], "VHS_VideoCombine")
        self.assertEqual(graph["extra"]["runpod_bundle"]["queue_jobs"], 10)
        self.assertEqual(graph["extra"]["runpod_bundle"]["auto_download"], "zip-after-final-job")

    def test_canvas_geometry_has_no_overlaps(self):
        def rect(n):
            x, y = n["pos"]
            w, h = n["size"]
            return x, y, x+w, y+h
        def bounds(g):
            x, y, w, h = g["bounding"]
            return x, y, x+w, y+h
        def overlap(a, b):
            return min(a[2],b[2])-max(a[0],b[0]) > .01 and min(a[3],b[3])-max(a[1],b[1]) > .01
        def contains(a, b):
            return b[0]>=a[0] and b[1]>=a[1] and b[2]<=a[2] and b[3]<=a[3]
        for path in WORKFLOWS:
            graph = load(path)
            nodes = graph["nodes"]
            groups = [g for g in graph["groups"] if g["id"] != 36]
            parent = next(g for g in graph["groups"] if g["id"] == 36)
            for i, first in enumerate(nodes):
                self.assertTrue(contains(bounds(parent), rect(first)))
                self.assertEqual(sum(contains(bounds(g),rect(first)) for g in groups), 1, first["id"])
                for second in nodes[i+1:]:
                    self.assertFalse(overlap(rect(first),rect(second)), (first["id"],second["id"]))
            for i, first in enumerate(groups):
                for second in groups[i+1:]:
                    self.assertFalse(overlap(bounds(first),bounds(second)), (first["id"],second["id"]))


if __name__ == "__main__":
    unittest.main()
