import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / "workflows" / "wan22_smooth_v6_aio_runpod.json",
    ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_runpod.json",
)


class WorkflowWiringTests(unittest.TestCase):
    def load(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_sampler_subgraphs_are_flattened(self):
        expected = {
            234: "KSamplerWithNAG (Advanced)",
            235: "KSamplerAdvanced",
            236: "KSamplerAdvanced",
            237: "KSamplerWithNAG (Advanced)",
            329: "KSamplerAdvanced",
            330: "KSamplerWithNAG (Advanced)",
        }
        for path in WORKFLOWS:
            with self.subTest(path=path.name):
                graph = self.load(path)
                by_id = {node["id"]: node for node in graph["nodes"]}
                self.assertEqual(
                    {node_id: by_id[node_id]["type"] for node_id in expected},
                    expected,
                )

    def test_nag_negative_fanout_is_preserved(self):
        for path in WORKFLOWS:
            with self.subTest(path=path.name):
                graph = self.load(path)
                by_id = {node["id"]: node for node in graph["nodes"]}
                links = {link[0]: link for link in graph["links"]}
                for node_id in (237, 330):
                    node = by_id[node_id]
                    inputs = {item["name"]: item for item in node["inputs"]}
                    negative = links[inputs["negative"]["link"]]
                    nag_negative = links[inputs["nag_negative"]["link"]]
                    self.assertNotEqual(negative[0], nag_negative[0])
                    self.assertEqual(negative[1:3], nag_negative[1:3])

    def test_every_top_level_link_matches_its_declared_slots(self):
        for path in WORKFLOWS:
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


if __name__ == "__main__":
    unittest.main()
