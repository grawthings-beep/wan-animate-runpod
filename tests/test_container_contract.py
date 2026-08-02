import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ContainerContractTests(unittest.TestCase):
    def test_base_image_is_pinned_by_amd64_manifest_digest(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        match = re.search(r"^ARG BASE_IMAGE=(\S+)$", dockerfile, re.MULTILINE)

        self.assertIsNotNone(match)
        self.assertRegex(match.group(1), r"^runpod/comfyui:1\.4\.4-cuda12\.8@sha256:[0-9a-f]{64}$")

    def test_cu128_runtime_contract_is_explicit(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("PIP_CONSTRAINT=/opt/comfyui-runtime-constraints.txt", dockerfile)
        self.assertIn("EXPECTED_TORCH_VERSION=2.10.0+cu128", dockerfile)
        self.assertIn("EXPECTED_TORCHVISION_VERSION=0.25.0+cu128", dockerfile)
        self.assertIn("EXPECTED_TORCHAUDIO_VERSION=2.10.0+cu128", dockerfile)
        self.assertIn("EXPECTED_TORCH_CUDA=12.8", dockerfile)

    def test_cuda_visibility_is_normalized_before_gpu_probe(self):
        start = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")

        normalize_at = start.index("normalize_cuda_visibility")
        probe_at = start.index("gpu_preflight.py")
        self.assertLess(normalize_at, probe_at)


if __name__ == "__main__":
    unittest.main()
