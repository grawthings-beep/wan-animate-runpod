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

    def test_bundled_loop_queue_nodes_are_copied_into_comfyui(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        package = ROOT / "custom_nodes" / "ComfyUI-WanLoopBatch"

        self.assertIn(
            "COPY custom_nodes/ /opt/comfyui-baked/custom_nodes/", dockerfile
        )
        self.assertTrue((package / "__init__.py").is_file())
        self.assertTrue((package / "web" / "wan_loop_batch.js").is_file())

    def test_auto_mosaic_runtime_is_pinned_and_cpu_node_is_bundled(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        package = ROOT / "custom_nodes" / "ComfyUI-WanLoopBatch"
        requirements = (package / "requirements.txt").read_text(encoding="utf-8")

        self.assertTrue((package / "mosaic_nodes.py").is_file())
        self.assertEqual(requirements.strip(), "ultralytics==8.4.104")
        self.assertIn("ComfyUI-WanLoopBatch/requirements.txt", dockerfile)

    def test_custom_nodes_use_retryable_commit_pinned_fetches(self):
        installer = (ROOT / "scripts" / "install_custom_nodes.sh").read_text(
            encoding="utf-8"
        )
        manifest = (ROOT / "custom_nodes.txt").read_text(encoding="utf-8")
        entries = [
            line.split("|")
            for line in manifest.splitlines()
            if line and not line.startswith("#")
        ]

        self.assertIn('GIT_FETCH_ATTEMPTS="${GIT_FETCH_ATTEMPTS:-5}"', installer)
        self.assertIn("fetch_pinned_node", installer)
        self.assertIn("http.version=HTTP/1.1", installer)
        self.assertIn("fetch --depth 1 origin", installer)
        self.assertNotIn("git clone", installer)
        self.assertTrue(entries)
        self.assertTrue(all(len(entry) == 3 for entry in entries))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", entry[2]) for entry in entries))

    def test_loop_image_uses_a_separate_minimal_node_manifest(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        full_manifest = (ROOT / "custom_nodes.txt").read_text(encoding="utf-8")
        loop_manifest = (ROOT / "custom_nodes.loop.txt").read_text(encoding="utf-8")

        def names(text):
            return {
                line.split("|", 1)[0]
                for line in text.splitlines()
                if line and not line.startswith("#")
            }

        loop_names = names(loop_manifest)
        full_names = names(full_manifest)
        self.assertIn("ARG CUSTOM_NODES_MANIFEST=custom_nodes.txt", dockerfile)
        self.assertIn(
            "COPY ${CUSTOM_NODES_MANIFEST} /opt/runpod-wan-animate/custom_nodes.txt",
            dockerfile,
        )
        self.assertTrue(loop_names < full_names)
        self.assertNotIn("ComfyUI-MMAudio", loop_names)
        self.assertNotIn("ComfyUI-WanVideoWrapper", loop_names)
        self.assertNotIn("ComfyUI-GGUF", loop_names)

    def test_image_build_removes_git_metadata_in_the_same_layer(self):
        installer = (ROOT / "scripts" / "install_custom_nodes.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('KEEP_CUSTOM_NODE_GIT:-0', installer)
        self.assertIn("-name .git", installer)


if __name__ == "__main__":
    unittest.main()
