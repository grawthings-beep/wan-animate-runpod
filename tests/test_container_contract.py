import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ContainerContractTests(unittest.TestCase):
    def setUp(self):
        self.loop = (ROOT / "Dockerfile.loop").read_text(encoding="utf-8")
        self.ci = (ROOT / ".github/workflows/build-ghcr.yml").read_text(
            encoding="utf-8"
        )

    def test_production_default_uses_pinned_official_amd64_base(self):
        self.assertRegex(
            self.loop,
            r"ARG BASE_IMAGE=runpod/comfyui:1\.4\.4-cuda12\.8@sha256:[0-9a-f]{64}",
        )
        self.assertNotIn("pytorch/pytorch:", self.loop)

    def test_ada_and_blackwell_stacks_are_separate_matrix_variants(self):
        self.assertIn("loop-ada-cu128", self.ci)
        self.assertIn("loop-blackwell-cu130", self.ci)
        self.assertIn("cuda12.8@sha256:7078f94d", self.ci)
        self.assertIn("cuda13.0@sha256:949b0688", self.ci)
        self.assertIn("2.10.0+cu128", self.ci)
        self.assertIn("2.10.0+cu130", self.ci)
        self.assertIn("max-parallel: 2", self.ci)

    def test_runtime_contract_is_parameterized_and_checked(self):
        for name in (
            "EXPECTED_TORCH_VERSION",
            "EXPECTED_TORCHVISION_VERSION",
            "EXPECTED_TORCHAUDIO_VERSION",
            "EXPECTED_TORCH_CUDA",
            "WAN_GPU_FAMILY",
        ):
            self.assertIn(f"ARG {name}=", self.loop)
            self.assertIn(f"{name}=${{{name}}}", self.loop)
        self.assertIn("gpu_preflight.py --stack-only", self.loop)

    def test_expensive_dependencies_precede_volatile_bundle_content(self):
        install_at = self.loop.index("install_custom_nodes.sh")
        workflows_at = self.loop.index("COPY workflows/")
        scripts_at = self.loop.index("COPY scripts/ /opt/runpod-wan-animate/scripts/")
        self.assertLess(install_at, workflows_at)
        self.assertLess(install_at, scripts_at)
        self.assertIn("cache-to: type=registry", self.ci)
        self.assertNotIn("type=gha", self.ci)

    def test_build_runs_real_entrypoint_smoke_test(self):
        smoke = (ROOT / "scripts/container_smoke.sh").read_text(encoding="utf-8")
        self.assertIn("/opt/runpod-wan-animate/scripts/start.sh", smoke)
        self.assertIn("--quick-test-for-ci", smoke)
        self.assertIn("--max-upload-size 300", smoke)
        self.assertIn("BOOTSTRAP_STATUS=0", smoke)
        self.assertIn("container_smoke.sh", self.loop)

    def test_batch_bulk_import_frontend_and_zip_route_are_bundled(self):
        package = ROOT / "custom_nodes" / "ComfyUI-WanLoopBatch"
        frontend = (package / "web" / "wan_loop_batch.js").read_text(
            encoding="utf-8"
        )
        parser = (package / "web" / "wan_loop_prompt_parser.mjs").read_text(
            encoding="utf-8"
        )
        routes = (package / "batch_routes.py").read_text(encoding="utf-8")
        self.assertIn("webkitGetAsEntry", frontend)
        self.assertIn("exactly 10 prompts", parser)
        self.assertIn("/wan-loop/batch/import-zip", frontend)
        self.assertIn("/wan-loop/batch/import-zip", routes)
        self.assertIn("extract_image_zip", routes)
        self.assertIn("blank-line-separated prompt blocks", parser)

    def test_database_and_boot_status_use_writable_workspace(self):
        start = (ROOT / "scripts/start.sh").read_text(encoding="utf-8")
        self.assertIn(
            'COMFYUI_DATABASE_URL="${COMFYUI_DATABASE_URL:-sqlite:///${WORKSPACE_DIR}/user/comfyui.db}"',
            start,
        )
        self.assertIn('--database-url "${COMFYUI_DATABASE_URL}"', start)
        self.assertIn("bootstrap_status.py serve", start)
        self.assertLess(start.index("bootstrap_status.py serve"), start.index("download_models.py"))

    def test_cuda_visibility_is_normalized_before_gpu_probe(self):
        start = (ROOT / "scripts/start.sh").read_text(encoding="utf-8")
        self.assertLess(start.index("normalize_cuda_visibility"), start.index("gpu_preflight.py"))

    def test_loop_image_uses_pinned_minimal_node_manifest(self):
        full = (ROOT / "custom_nodes.txt").read_text(encoding="utf-8")
        loop = (ROOT / "custom_nodes.loop.txt").read_text(encoding="utf-8")

        def names(text):
            return {
                line.split("|", 1)[0]
                for line in text.splitlines()
                if line and not line.startswith("#")
            }

        self.assertTrue(names(loop) < names(full))
        self.assertNotIn("ComfyUI-MMAudio", names(loop))
        self.assertIn("COPY custom_nodes.loop.txt", self.loop)

    def test_custom_node_fetches_are_retryable_and_commit_pinned(self):
        installer = (ROOT / "scripts/install_custom_nodes.sh").read_text(
            encoding="utf-8"
        )
        entries = [
            line.split("|")
            for line in (ROOT / "custom_nodes.loop.txt").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        self.assertIn("fetch_pinned_node", installer)
        self.assertIn("http.version=HTTP/1.1", installer)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", entry[2]) for entry in entries))

    def test_legacy_full_build_is_manual_only(self):
        full_ci = (ROOT / ".github/workflows/build-full.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("workflow_dispatch:", full_ci)
        self.assertNotRegex(full_ci, r"(?m)^  push:")
        self.assertNotIn("Dockerfile\n", self.ci)


if __name__ == "__main__":
    unittest.main()
