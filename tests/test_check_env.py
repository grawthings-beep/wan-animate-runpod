import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_env", ROOT / "scripts" / "check_env.py"
)
CHECK_ENV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_ENV)


class CheckEnvTests(unittest.TestCase):
    def test_optional_missing_asset_does_not_count_as_required(self):
        entries = [
            {"path": "models/required.bin", "required": True},
            {"path": "models/optional.bin", "required": False},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            required, optional = CHECK_ENV.missing_by_requirement(root, entries)

        self.assertEqual(required, [root / "models/required.bin"])
        self.assertEqual(optional, [root / "models/optional.bin"])

    def test_existing_required_asset_is_not_missing(self):
        entries = [{"path": "models/required.bin"}]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path = root / "models/required.bin"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"ok")
            required, optional = CHECK_ENV.missing_by_requirement(root, entries)

        self.assertEqual(required, [])
        self.assertEqual(optional, [])

    def test_strict_mode_allows_missing_optional_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "profiles": {"test": {"include_groups": ["optional"]}},
                        "models": [
                            {
                                "name": "optional",
                                "group": "optional",
                                "path": "models/optional.bin",
                                "required": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = CHECK_ENV.main(
                [
                    "--manifest",
                    str(manifest),
                    "--profile",
                    "test",
                    "--model-root",
                    str(root),
                    "--strict",
                ]
            )

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
