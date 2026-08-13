import importlib.util
import pathlib
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpu_preflight", ROOT / "scripts" / "gpu_preflight.py"
)
GPU_PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GPU_PREFLIGHT)


class GpuPreflightTests(unittest.TestCase):
    def test_pinned_torch_stack_is_required(self):
        torch_stack = mock.Mock(
            returncode=1,
            stdout="",
            stderr="RuntimeError: incompatible pinned torch stack",
        )

        result = GPU_PREFLIGHT.probe_torch_stack("python", runner=mock.Mock(return_value=torch_stack))

        self.assertFalse(result.ready)
        self.assertFalse(result.retryable)
        self.assertIn("incompatible pinned torch stack", result.diagnostic)

    def test_real_cuda_operation_is_required(self):
        nvidia = mock.Mock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 5090, GPU-123, 32607 MiB, 590.48.01\n",
            stderr="",
        )
        torch_probe = mock.Mock(
            returncode=1,
            stdout="",
            stderr="RuntimeError: CUDA unknown error",
        )
        runner = mock.Mock(side_effect=[nvidia, torch_probe])

        ready, diagnostic = GPU_PREFLIGHT.probe_once("python", runner=runner)

        self.assertFalse(ready)
        self.assertIn("nvidia-smi", diagnostic)
        self.assertIn("CUDA unknown error", diagnostic)

    def test_success_reports_gpu_and_torch_probe(self):
        nvidia = mock.Mock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090, GPU-456, 24564 MiB, 590.48.01\n",
            stderr="",
        )
        torch_probe = mock.Mock(
            returncode=0,
            stdout='{"torch": "2.8.0", "torch_cuda": "12.8", "device": "RTX 4090"}\n',
            stderr="",
        )
        runner = mock.Mock(side_effect=[nvidia, torch_probe])

        ready, diagnostic = GPU_PREFLIGHT.probe_once("python", runner=runner)

        self.assertTrue(ready)
        self.assertIn("RTX 4090", diagnostic)
        self.assertIn('"torch_cuda": "12.8"', diagnostic)

    def test_old_5090_driver_is_rejected_without_retry(self):
        nvidia = mock.Mock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 5090, GPU-123, 32607 MiB, 565.77\n",
            stderr="",
        )
        runner = mock.Mock(return_value=nvidia)

        with mock.patch.dict(
            GPU_PREFLIGHT.os.environ,
            {"EXPECTED_TORCH_CUDA": "13.0", "WAN_GPU_FAMILY": "blackwell"},
        ):
            result = GPU_PREFLIGHT.probe_once("python", runner=runner)

        self.assertFalse(result.ready)
        self.assertFalse(result.retryable)
        self.assertIn("incompatible CUDA 13 driver", result.diagnostic)
        runner.assert_called_once()

    def test_current_5090_driver_reaches_real_cuda_probe(self):
        nvidia = mock.Mock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 5090, GPU-123, 32607 MiB, 580.159.03\n",
            stderr="",
        )
        torch_probe = mock.Mock(
            returncode=0,
            stdout='{"torch": "2.10.0+cu128", "torch_cuda": "12.8", "device": "RTX 5090"}\n',
            stderr="",
        )
        runner = mock.Mock(side_effect=[nvidia, torch_probe])

        result = GPU_PREFLIGHT.probe_once("python", runner=runner)

        self.assertTrue(result.ready)
        self.assertTrue(result.retryable)
        self.assertEqual(runner.call_count, 2)

    def test_ada_image_rejects_blackwell_before_download(self):
        nvidia = mock.Mock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 5090, GPU-123, 32607 MiB, 590.48.01\n",
            stderr="",
        )
        runner = mock.Mock(return_value=nvidia)
        with mock.patch.dict(GPU_PREFLIGHT.os.environ, {"WAN_GPU_FAMILY": "ada"}):
            result = GPU_PREFLIGHT.probe_once("python", runner=runner)
        self.assertFalse(result.ready)
        self.assertFalse(result.retryable)
        self.assertIn("loop-blackwell-cu130", result.diagnostic)
        runner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
