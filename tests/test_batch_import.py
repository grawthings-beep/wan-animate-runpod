import base64
import importlib.util
import pathlib
import tempfile
import unittest
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "custom_nodes" / "ComfyUI-WanLoopBatch" / "batch_import.py"
)
SPEC = importlib.util.spec_from_file_location("wan_loop_batch_import", MODULE_PATH)
BATCH_IMPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH_IMPORT)

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class BatchImportTests(unittest.TestCase):
    def make_zip(self, root, count=10, payload=PNG_1X1, prompts=True):
        archive = root / "batch.zip"
        names = ["image-10.png", *[f"image-{number}.png" for number in range(1, 10)]]
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in names[:count]:
                bundle.writestr(f"../unsafe-folder/{name}", payload)
            if prompts:
                bundle.writestr(
                    "prompts.txt",
                    "\n".join(f"video prompt {number}" for number in range(1, 11)),
                )
        return archive

    def test_zip_is_flattened_and_naturally_sorted_into_unique_input_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            result = BATCH_IMPORT.extract_image_zip(
                self.make_zip(root), root / "input"
            )
            self.assertEqual(len(result["images"]), 10)
            self.assertTrue(result["images"][0]["display_name"].endswith("image-1.png"))
            self.assertTrue(result["images"][9]["display_name"].endswith("image-10.png"))
            self.assertIn("video prompt 10", result["prompts_text"])
            for index, item in enumerate(result["images"], start=1):
                self.assertNotIn("..", item["name"])
                self.assertIn(f"slot-{index:02d}-", item["name"])
                self.assertTrue((root / "input" / pathlib.PurePosixPath(item["name"])).is_file())

    def test_zip_must_contain_exactly_ten_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with self.assertRaisesRegex(ValueError, "exactly 10 images; found 9"):
                BATCH_IMPORT.extract_image_zip(
                    self.make_zip(root, count=9), root / "input"
                )

    def test_disguised_non_image_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with self.assertRaisesRegex(ValueError, "invalid image payload"):
                BATCH_IMPORT.extract_image_zip(
                    self.make_zip(root, payload=b"not really a png"), root / "input"
                )


if __name__ == "__main__":
    unittest.main()
