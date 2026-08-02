import importlib.util
import json
import pathlib
import tempfile
import unittest
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILS_PATH = (
    ROOT / "custom_nodes" / "ComfyUI-WanLoopBatch" / "batch_utils.py"
)
SPEC = importlib.util.spec_from_file_location("wan_loop_batch_utils", UTILS_PATH)
BATCH_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH_UTILS)


class WanLoopBatchArchiveTests(unittest.TestCase):
    def make_video(self, output_root, batch_id, slot):
        batch_dir = output_root / "Video" / "loop-batches" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        preview = batch_dir / f"slot-{slot:02d}_00001.png"
        video = batch_dir / f"slot-{slot:02d}_00001.mp4"
        preview.write_bytes(b"preview")
        video.write_bytes(f"video-{slot}".encode())
        return True, [str(preview), str(video)]

    def test_archive_is_created_only_after_slot_ten(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = pathlib.Path(directory)
            batch_id = "loop10-test"
            for slot in range(1, 11):
                result = BATCH_UTILS.record_video_and_maybe_archive(
                    output_root,
                    batch_id,
                    slot,
                    f"image-{slot:02d}.png",
                    f"prompt {slot:02d}",
                    self.make_video(output_root, batch_id, slot),
                    10,
                )
                if slot < 10:
                    self.assertIsNone(result["archive"])

            archive = result["archive"]
            self.assertTrue(archive.is_file())
            with zipfile.ZipFile(archive) as bundle:
                expected = [f"slot-{slot:02d}.mp4" for slot in range(1, 11)]
                self.assertEqual(bundle.namelist(), [*expected, "manifest.json"])
                manifest = json.loads(bundle.read("manifest.json"))
                self.assertEqual(manifest["expected_count"], 10)
                self.assertEqual(manifest["videos"]["7"]["archive_file"], "slot-07.mp4")
                self.assertEqual(manifest["videos"]["7"]["positive_prompt"], "prompt 07")

    def test_final_slot_refuses_an_incomplete_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = pathlib.Path(directory)
            batch_id = "loop10-incomplete"
            with self.assertRaisesRegex(RuntimeError, "earlier slots are missing"):
                BATCH_UTILS.record_video_and_maybe_archive(
                    output_root,
                    batch_id,
                    10,
                    "image-10.png",
                    "prompt 10",
                    self.make_video(output_root, batch_id, 10),
                    10,
                )
            archive = (
                output_root
                / "Video"
                / "loop-batches"
                / batch_id
                / f"{batch_id}.zip"
            )
            self.assertFalse(archive.exists())

    def test_video_must_be_inside_its_exact_batch_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = pathlib.Path(directory)
            outside = output_root / "outside.mp4"
            outside.write_bytes(b"video")
            with self.assertRaisesRegex(ValueError, "escapes batch directory"):
                BATCH_UTILS.record_video_and_maybe_archive(
                    output_root,
                    "loop10-safe",
                    1,
                    "image.png",
                    "prompt",
                    (True, [str(outside)]),
                    10,
                )

    def test_batch_id_rejects_path_characters(self):
        for value in ("../escape", "with/slash", "", "a" * 81):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    BATCH_UTILS.validate_batch_id(value)


if __name__ == "__main__":
    unittest.main()
