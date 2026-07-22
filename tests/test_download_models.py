import hashlib
import importlib.util
import pathlib
import tempfile
import unittest
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "download_models", ROOT / "scripts" / "download_models.py"
)
DOWNLOAD_MODELS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD_MODELS)


class DownloadModelsTests(unittest.TestCase):
    def test_existing_file_gets_verified_marker(self):
        payload = b"verified model fixture"
        expected_sha = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "model.bin"
            output.write_bytes(payload)
            self.assertTrue(
                DOWNLOAD_MODELS.valid_existing(
                    output, expected_sha, len(payload), len(payload)
                )
            )
            self.assertTrue(DOWNLOAD_MODELS.marker_path(output).is_file())

    def test_wrong_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "model.bin"
            output.write_bytes(b"too short")
            self.assertFalse(DOWNLOAD_MODELS.valid_existing(output, "", 99, 1))

    def test_zip_extraction_is_flattened_and_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            archive = root / "model.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("nested/flownet.pkl", b"weights")
                bundle.writestr("nested/readme.txt", b"ignore")
            extracted = DOWNLOAD_MODELS.extract_archive(archive, root, "pkl")
            self.assertEqual([path.name for path in extracted], ["flownet.pkl"])
            self.assertEqual((root / "flownet.pkl").read_bytes(), b"weights")
            self.assertFalse(archive.exists())

    def test_profile_groups_are_exact(self):
        manifest = {
            "profiles": {
                "quality": {"include_groups": ["shared", "i2v"]},
            }
        }
        self.assertEqual(
            DOWNLOAD_MODELS.selected_groups(manifest, "quality"), {"shared", "i2v"}
        )
        with self.assertRaises(ValueError):
            DOWNLOAD_MODELS.selected_groups(manifest, "missing")


if __name__ == "__main__":
    unittest.main()
