import hashlib
import importlib.util
import os
import pathlib
import tempfile
import unittest
import zipfile
from collections import namedtuple
from unittest import mock


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

    def test_huggingface_resolve_url_is_parsed(self):
        self.assertEqual(
            DOWNLOAD_MODELS.parse_huggingface_url(
                "https://huggingface.co/org/repo/resolve/abc123/folder/model.safetensors"
            ),
            ("org/repo", "abc123", "folder/model.safetensors"),
        )
        self.assertIsNone(
            DOWNLOAD_MODELS.parse_huggingface_url("https://civitai.com/api/download/models/1")
        )
        self.assertEqual(
            DOWNLOAD_MODELS.parse_huggingface_url(
                "https://huggingface.co/org/repo/resolve/refs%2Fpr%2F1/model.bin"
            ),
            ("org/repo", "refs/pr/1", "model.bin"),
        )
        self.assertEqual(
            DOWNLOAD_MODELS.parse_huggingface_url(
                "https://huggingface.co/org/repo/resolve/main/model%20(2).safetensors"
            ),
            ("org/repo", "main", "model (2).safetensors"),
        )

    def test_largest_files_are_scheduled_first(self):
        entries = [
            {"name": "small", "size_bytes": 1},
            {"name": "large", "size_bytes": 100},
            {"name": "medium", "size_bytes": 10},
        ]
        self.assertEqual(
            [item["name"] for item in DOWNLOAD_MODELS.largest_first(entries)],
            ["large", "medium", "small"],
        )

    def test_cached_file_is_materialized_without_changing_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            cached = root / "cache" / "blob"
            cached.parent.mkdir()
            cached.write_bytes(b"xet cache payload")
            part = root / "models" / "model.part"
            DOWNLOAD_MODELS.materialize_cached_file(cached, part)
            self.assertEqual(part.read_bytes(), b"xet cache payload")

    def test_disk_preflight_accounts_for_resumable_partial(self):
        entries = [
            {
                "name": "large",
                "path": "models/large.bin",
                "size_bytes": 10,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            part = root / "models" / "large.bin.part"
            part.parent.mkdir()
            part.write_bytes(b"1234")
            missing, unknown = DOWNLOAD_MODELS.missing_download_bytes(entries, root)
        self.assertEqual(missing, 6)
        self.assertEqual(unknown, [])

    def test_disk_preflight_fails_before_an_undersized_download(self):
        usage = namedtuple("usage", "total used free")
        entries = [
            {
                "name": "model",
                "path": "models/model.bin",
                "size_bytes": 40_000_000_000,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Increase the RunPod Volume Disk"):
                DOWNLOAD_MODELS.ensure_disk_capacity(
                    entries,
                    root,
                    headroom_gb=12,
                    disk_usage=lambda _path: usage(
                        50_000_000_000, 5_000_000_000, 45_000_000_000
                    ),
                )

    def test_disk_preflight_accepts_sufficient_capacity(self):
        usage = namedtuple("usage", "total used free")
        entries = [
            {
                "name": "model",
                "path": "models/model.bin",
                "size_bytes": 40_000_000_000,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            DOWNLOAD_MODELS.ensure_disk_capacity(
                entries,
                pathlib.Path(directory),
                headroom_gb=12,
                disk_usage=lambda _path: usage(
                    100_000_000_000, 10_000_000_000, 90_000_000_000
                ),
            )

    def test_auth_query_is_added_without_losing_existing_parameters(self):
        with mock.patch.dict(os.environ, {"CIVITAI_API_TOKEN": "top secret"}):
            url = DOWNLOAD_MODELS.add_auth_query(
                "https://civitai.com/api/download/models/1?format=SafeTensor",
                "CIVITAI_API_TOKEN",
            )
        self.assertIn("format=SafeTensor", url)
        self.assertIn("token=top+secret", url)

    def test_auth_query_is_redacted_from_logs(self):
        url = "https://civitai.com/api/download/models/1?token=top-secret&format=SafeTensor"
        redacted = DOWNLOAD_MODELS.redact_url(url)
        self.assertNotIn("top-secret", redacted)
        self.assertIn("token=REDACTED", redacted)

    def test_loop_profile_contains_only_connected_loop_asset_groups(self):
        import json

        manifest = json.loads(
            (ROOT / "config" / "wan22-models.json").read_text(encoding="utf-8")
        )
        groups = DOWNLOAD_MODELS.selected_groups(manifest, "loop-quality")
        self.assertEqual(
            groups,
            {
                "text-common",
                "clip-vision",
                "vae-fp32",
                "i2v",
                "lightx-i2v",
                "loop-nsfw-loras",
                "rife49",
            },
        )
        selected = {
            pathlib.PurePosixPath(entry["path"]).name
            for entry in manifest["models"]
            if entry["group"] in groups
        }
        self.assertEqual(len(selected), 9)
        self.assertIn(
            "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            selected,
        )
        self.assertFalse(any(name.startswith("mmaudio_") for name in selected))
        self.assertFalse(any(name.endswith(".gguf") for name in selected))
        self.assertTrue(
            {"NSFW-22-H-e8.safetensors", "NSFW-22-L-e8.safetensors"}.issubset(
                selected
            )
        )

    def test_loop_workflow_declares_compatible_profile(self):
        import json

        workflow = json.loads(
            (ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_runpod.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            workflow["extra"]["runpod_bundle"]["profile"], "loop-quality"
        )

    def test_lightning_profile_contains_every_workflow_asset(self):
        import json

        manifest = json.loads(
            (ROOT / "config" / "wan22-models.json").read_text(encoding="utf-8")
        )
        workflow = json.loads(
            (
                ROOT
                / "workflows"
                / "wan22_native_enhanced_lightning_longvideo_runpod.json"
            ).read_text(encoding="utf-8")
        )
        profile = workflow["extra"]["runpod_bundle"]["profile"]
        self.assertEqual(profile, "lightning-longvideo")
        groups = DOWNLOAD_MODELS.selected_groups(manifest, profile)
        selected = {
            pathlib.PurePosixPath(entry["path"]).name
            for entry in manifest["models"]
            if entry["group"] in groups
        }
        self.assertTrue(
            {
                "wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2Q8H.gguf",
                "wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2Q8L.gguf",
                "SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors",
                "SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors",
                "Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
                "4x_NMKD-Siax_200k.pth",
                "rife49.pth",
            }.issubset(selected)
        )
        self.assertNotIn(
            "wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2FP8H.safetensors",
            selected,
        )
        self.assertNotIn(
            "wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2FP8L.safetensors",
            selected,
        )

    def test_lightning_fp8_extras_are_optional(self):
        import json

        manifest = json.loads(
            (ROOT / "config" / "wan22-models.json").read_text(encoding="utf-8")
        )
        fp8 = [
            entry
            for entry in manifest["models"]
            if entry.get("group") == "lightning-native-fp8-extras"
        ]
        self.assertEqual(len(fp8), 2)
        self.assertTrue(all(entry.get("required") is False for entry in fp8))


if __name__ == "__main__":
    unittest.main()
