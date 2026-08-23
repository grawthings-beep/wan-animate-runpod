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
        groups = DOWNLOAD_MODELS.selected_groups(manifest, "loop-all")
        self.assertEqual(
            groups,
            {
                "text-common",
                "clip-vision",
                "vae-fp32",
                "i2v",
                "lightx-i2v",
                "loop-nsfw-loras",
                "loop-xxx-loras",
                "loop-cumshot-loras",
                "loop-joi-loras",
                "loop-throat-loras",
                "loop-iroiro-high-loras",
                "loop-iroiro-low-loras",
                "rife49",
                "auto-mosaic",
                "upscale-nmkd",
            },
        )
        selected = {
            pathlib.PurePosixPath(entry["path"]).name
            for entry in manifest["models"]
            if entry["group"] in groups
        }
        self.assertEqual(len(selected), 29)
        self.assertIn("4x_NMKD-Siax_200k.pth", selected)
        self.assertIn("animeNSFWDetection_v50.zip", selected)
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
        self.assertTrue(
            {
                "SmoothXXXAnimation_High.safetensors",
                "SmoothXXXAnimation_Low.safetensors",
            }.issubset(selected)
        )
        self.assertTrue(
            {
                "Cumshot_Aesthetics_High.safetensors",
                "Cumshot_Aesthetics_Low.safetensors",
            }.issubset(selected)
        )
        self.assertTrue(
            {
                "I2V_joi_trend_high.safetensors",
                "I2V_joi_trend_low.safetensors",
            }.issubset(selected)
        )
        self.assertTrue(
            {
                "Wan22_ThroatV3_High.safetensors",
                "Wan22_ThroatV3_Low.safetensors",
            }.issubset(selected)
        )
        self.assertTrue(
            {
                "cheek_bulge_fellatio_high_wan-2-2_i2v_A14B.safetensors",
                "glans_licking_high_wan-2-2_i2v_A14B.safetensors",
                "head_back_high_wan-2-2_i2v_A14B.safetensors",
                "paizuri_unaligned_breasts_high_wan-2-2_i2v_A14B.safetensors",
                "washizukami_high_wan-2-2_i2v_A14B.safetensors",
            }.issubset(selected)
        )
        self.assertTrue(
            {
                "cheek_bulge_fellatio_wanvideo_i2v.safetensors",
                "glans_licking_wanvideo_i2v_epoch5.safetensors",
                "head_back_wanvideo_i2v_epoch5.safetensors",
                "paizuri_unaligned_breasts_wanvideo_i2v_epoch5.safetensors",
                "washizukami_wanvideo_i2v.safetensors",
            }.issubset(selected)
        )

    def test_loop_xxx_loras_use_hugging_face_backup(self):
        import json

        manifest = json.loads(
            (ROOT / "config" / "wan22-models.json").read_text(encoding="utf-8")
        )
        entries = [
            entry for entry in manifest["models"] if entry["group"] == "loop-xxx-loras"
        ]
        self.assertEqual(len(entries), 2)
        self.assertTrue(
            all(
                entry["url"].startswith(
                    "https://huggingface.co/uwgm/nikke-civitai-backup/resolve/main/"
                )
                for entry in entries
            )
        )
        self.assertTrue(all(entry.get("requires_env") == ["HF_TOKEN"] for entry in entries))

    def test_loop_cumshot_loras_are_verified_hugging_face_downloads(self):
        import json

        manifest = json.loads(
            (ROOT / "config" / "wan22-models.json").read_text(encoding="utf-8")
        )
        entries = [
            entry
            for entry in manifest["models"]
            if entry["group"] == "loop-cumshot-loras"
        ]
        self.assertEqual(len(entries), 2)
        self.assertEqual(
            {entry["size_bytes"] for entry in entries}, {306807976}
        )
        self.assertEqual(
            {entry["sha256"] for entry in entries},
            {
                "a63d6ed7bca18ed83dd60a7f42951f7809ea718bfef52cf6ecec204c619eb218",
                "73603d65fa727b99b101bfc66cf6c9c5c2610c18394aa4ff1db4f096633aa627",
            },
        )
        self.assertTrue(
            all(
                entry["url"].startswith(
                    "https://huggingface.co/uwgm/nikke-civitai-backup/resolve/main/"
                )
                for entry in entries
            )
        )
        self.assertTrue(
            all(entry.get("requires_env") == ["HF_TOKEN"] for entry in entries)
        )

    def test_loop_joi_loras_are_verified_civitai_downloads(self):
        import json

        manifest = json.loads(
            (ROOT / "config" / "wan22-models.json").read_text(encoding="utf-8")
        )
        entries = [
            entry
            for entry in manifest["models"]
            if entry["group"] == "loop-joi-loras"
        ]
        self.assertEqual(len(entries), 2)
        self.assertEqual({entry["size_bytes"] for entry in entries}, {613516752})
        self.assertEqual(
            {entry["sha256"] for entry in entries},
            {
                "e625ad701cc0af550c823b822fe6025348f4a9cf4d3106275f295eb779b5979b",
                "71a1e1992b1092feef51d083309c4dd72c8fcc25cf007d9d818b9b3ce62c363c",
            },
        )
        self.assertEqual(
            {entry["url"] for entry in entries},
            {
                "https://civitai.red/api/download/models/2206435?fileId=2099355",
                "https://civitai.red/api/download/models/2206446?fileId=2099364",
            },
        )
        self.assertTrue(
            all(
                entry.get("requires_env") == ["CIVITAI_API_TOKEN"]
                and entry.get("auth_query_env") == "CIVITAI_API_TOKEN"
                for entry in entries
            )
        )

    def test_loop_iroiro_loras_are_public_revision_pinned_downloads(self):
        import json

        manifest = json.loads(
            (ROOT / "config" / "wan22-models.json").read_text(encoding="utf-8")
        )
        entries = [
            entry
            for entry in manifest["models"]
            if entry["group"] == "loop-iroiro-high-loras"
        ]
        self.assertEqual(len(entries), 5)
        self.assertEqual({entry["size_bytes"] for entry in entries}, {306807976})
        self.assertEqual(
            {entry["sha256"] for entry in entries},
            {
                "8e224b68e3eff0a037a925772df0db26ac3483f30247b0992ebbb61213b2fa78",
                "89fd30a5c977c01a8f3e6f531d1fcb36842ef12088e4aaeb7b45f46cfcd3db4d",
                "a6a4f0a470d84ae328abcb430204d068e0d3a2420e83960fe2f6f0992f422b58",
                "5183a314ba80d490227805299fdc671b9fdabf3f030fd2388c405fae8693189b",
                "07587222bdf99b026000490002f30cb39a84bea0c7e9ada774884033774ecee9",
            },
        )
        pinned_prefix = (
            "https://huggingface.co/nashikone/iroiroLoRA/resolve/"
            "bb185a26e882aefc5e7473bbef9340ad3ab1b1da/"
        )
        self.assertTrue(all(entry["url"].startswith(pinned_prefix) for entry in entries))
        self.assertTrue(all("requires_env" not in entry for entry in entries))

    def test_loop_throat_loras_are_verified_civitai_downloads(self):
        import json

        manifest = json.loads(
            (ROOT / "config" / "wan22-models.json").read_text(encoding="utf-8")
        )
        entries = [
            entry
            for entry in manifest["models"]
            if entry["group"] == "loop-throat-loras"
        ]
        self.assertEqual(len(entries), 2)
        self.assertEqual(
            {entry["size_bytes"] for entry in entries},
            {306831680, 306831672},
        )
        self.assertEqual(
            {entry["sha256"] for entry in entries},
            {
                "05331437178430556a859b3c136ca35be5f08327848cfa66e0e983b1bc0f4ebb",
                "17f7290bed73020bf9d5aaa8abb9ee4024606184dc2ab963047b927406d8ad9a",
            },
        )
        self.assertEqual(
            {entry["url"] for entry in entries},
            {
                "https://civitai.red/api/download/models/2517513?fileId=2405261",
                "https://civitai.red/api/download/models/2517548?fileId=2405303",
            },
        )
        self.assertTrue(
            all(
                entry.get("requires_env") == ["CIVITAI_API_TOKEN"]
                and entry.get("auth_query_env") == "CIVITAI_API_TOKEN"
                for entry in entries
            )
        )

    def test_loop_iroiro_low_loras_are_verified_revision_pinned_downloads(self):
        import json

        manifest = json.loads(
            (ROOT / "config" / "wan22-models.json").read_text(encoding="utf-8")
        )
        entries = [
            entry
            for entry in manifest["models"]
            if entry["group"] == "loop-iroiro-low-loras"
        ]
        self.assertEqual(len(entries), 5)
        self.assertEqual(
            {entry["size_bytes"] for entry in entries},
            {359258496, 359258504, 359258528, 359258552},
        )
        self.assertEqual(
            {entry["sha256"] for entry in entries},
            {
                "9c94c2abe3fdd8c7ef7b23907c4c1c11f607d11065237b7ceb5fd41f5e887837",
                "fa7a51365d12d37cd195a01e560f3629a3aa91a61dc54f0b4534f8c490ff8de2",
                "aca07c8f4e2937daf73b7ad17f8ad7c913cef30a78e7113fdb34e7c73b527621",
                "4f44476da0017942fa431906559bba959e63640ea61859758bb8ebe4f338b632",
                "ae5e93695ca512b27c387fcfd9b79fce6c15b008b961bca6f60071aec35bab53",
            },
        )
        pinned_prefix = (
            "https://huggingface.co/nashikone/iroiroLoRA/resolve/"
            "bb185a26e882aefc5e7473bbef9340ad3ab1b1da/"
            "Wan2.1_i2v_720p_14B_fp16/"
        )
        self.assertTrue(all(entry["url"].startswith(pinned_prefix) for entry in entries))
        self.assertTrue(all("requires_env" not in entry for entry in entries))

    def test_loop_workflow_declares_compatible_profile(self):
        import json

        workflow = json.loads(
            (ROOT / "workflows" / "wan22_smooth_v6_seamless_loop_runpod.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            workflow["extra"]["runpod_bundle"]["profile"], "loop-all"
        )

    def test_loop_core_omits_disabled_optional_lora_groups(self):
        import json

        manifest = json.loads(
            (ROOT / "config" / "wan22-models.json").read_text(encoding="utf-8")
        )
        groups = DOWNLOAD_MODELS.selected_groups(manifest, "loop-core")
        self.assertEqual(
            groups,
            {
                "text-common",
                "clip-vision",
                "vae-fp32",
                "i2v",
                "lightx-i2v",
                "loop-nsfw-loras",
                "loop-xxx-loras",
                "rife49",
                "auto-mosaic",
                "upscale-nmkd",
            },
        )

    def test_auto_mosaic_segmentation_archive_is_pinned_and_verified(self):
        import json

        manifest = json.loads(
            (ROOT / "config" / "wan22-models.json").read_text(encoding="utf-8")
        )
        entries = [
            entry for entry in manifest["models"] if entry["group"] == "auto-mosaic"
        ]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["size_bytes"], 18846815)
        self.assertEqual(
            entry["sha256"],
            "aca92864d30384b8dd7851b32e7ade621a147730bf9710fb4417214e0c61d690",
        )
        self.assertEqual(
            entry["url"], "https://civitai.com/api/download/models/2266294"
        )
        self.assertEqual(entry["extract"], "pt")
        self.assertEqual(
            entry["provides"],
            ["models/auto_mosaic/ntd11_anime_nsfw_segm_v5.pt"],
        )
        self.assertEqual(entry["requires_env"], ["CIVITAI_API_TOKEN"])
        self.assertEqual(entry["auth_query_env"], "CIVITAI_API_TOKEN")

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
