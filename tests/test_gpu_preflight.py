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


if __name__ == "__main__":
    unittest.main()
