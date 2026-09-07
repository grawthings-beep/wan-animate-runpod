import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("workflow_install", ROOT / "scripts/install_workflows.py")
INSTALL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(INSTALL)
MANIFEST = json.loads((ROOT / "config/wan22-models.json").read_text(encoding="utf-8"))


class WorkflowInstallTests(unittest.TestCase):
    def run_install(self, root, profile="loop-all"):
        INSTALL.install(ROOT / "workflows", root / "ui", root / "backup", MANIFEST, profile)

    def test_fresh_install_and_repeat_have_exactly_two_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.run_install(root)
            first = {p.name: p.read_bytes() for p in (root / "ui").iterdir()}
            self.run_install(root)
            self.assertEqual(set(first), set(INSTALL.CURRENT_NAMES))
            self.assertEqual(first, {p.name: p.read_bytes() for p in (root / "ui").iterdir()})
            self.assertEqual(list((root / "backup").iterdir()), [])

    def test_old_and_edited_copies_are_preserved_but_not_left_in_ui(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "ui").mkdir()
            old = "wan22_smooth_v6_seamless_loop_core_auto_mosaic_runpod"
            (root / "ui" / f"{old}.json").write_bytes(b"my edited preset")
            (root / "ui" / f"{old}-bundle-abcdef123456.json").write_bytes(b"older bundle")
            (root / "ui" / INSTALL.CURRENT_NAMES[0]).write_bytes(b"edited current")
            (root / "ui" / "my-custom.json").write_bytes(b"do not touch")
            self.run_install(root)
            self.assertEqual({p.read_bytes() for p in (root / "backup").iterdir()},
                             {b"my edited preset", b"older bundle", b"edited current"})
            self.assertEqual({p.name for p in (root / "ui").iterdir()},
                             {*INSTALL.CURRENT_NAMES, "my-custom.json"})
            self.assertEqual((root / "ui/my-custom.json").read_bytes(), b"do not touch")
            self.run_install(root)
            self.assertEqual(len(list((root / "backup").iterdir())), 3)

    def test_core_profile_keeps_same_two_names_and_only_available_lora_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.run_install(root, "loop-core")
            self.assertEqual({p.name for p in (root / "ui").iterdir()}, set(INSTALL.CURRENT_NAMES))
            for path in (root / "ui").iterdir():
                graph = json.loads(path.read_text(encoding="utf-8"))
                names = [w["lora"] for n in graph["nodes"]
                         if n["type"] == "Power Lora Loader (rgthree)"
                         for w in n["widgets_values"] if isinstance(w, dict) and w.get("lora")]
                self.assertEqual(len(names), 5)
                self.assertIn("wind.safetensors", names)
                self.assertEqual(graph["extra"]["runpod_bundle"]["profile"], "loop-core")

    def test_does_not_match_similarly_named_user_files(self):
        self.assertFalse(INSTALL.is_managed_name("my-wan22_loop_single_runpod.json"))
        self.assertFalse(INSTALL.is_managed_name("wan22_loop_single_runpod-custom.json"))

    def test_backup_cannot_be_inside_workflow_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with self.assertRaises(ValueError):
                INSTALL.install(ROOT / "workflows", root / "ui", root / "ui/backup", MANIFEST, "loop-all")


if __name__ == "__main__":
    unittest.main()
