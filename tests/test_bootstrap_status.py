import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bootstrap_status", ROOT / "scripts" / "bootstrap_status.py"
)
STATUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATUS)


class BootstrapStatusTests(unittest.TestCase):
    def test_atomic_updates_preserve_progress_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "status.json"
            STATUS.write_status(path, {"assets_total": 12, "state": "initializing"})
            STATUS.write_status(path, {"assets_completed": 3, "phase": "models"})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["assets_total"], 12)
            self.assertEqual(data["assets_completed"], 3)
            self.assertEqual(data["phase"], "models")
            self.assertIsInstance(data["updated_at"], int)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_page_escapes_status_content_and_polls_handoff(self):
        page = STATUS.render_page(
            {"state": "initializing", "phase": "models", "message": "<unsafe>"}
        )
        self.assertIn("&lt;unsafe&gt;", page)
        self.assertNotIn("<unsafe>", page)
        self.assertIn("/status.json", page)
        self.assertIn("location.reload()", page)
        self.assertIn("準備が終わるまで", page)


if __name__ == "__main__":
    unittest.main()
