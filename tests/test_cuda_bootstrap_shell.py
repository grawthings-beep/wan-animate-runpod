import pathlib
import shutil
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


@unittest.skipUnless(BASH, "bash is required for startup shell tests")
class CudaBootstrapShellTests(unittest.TestCase):
    def run_shell(self, body):
        result = subprocess.run(
            [BASH, "-c", f"source scripts/common.sh\n{body}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_single_gpu_is_normalized_to_zero_before_torch(self):
        output = self.run_shell(
            r'''
nvidia-smi() {
  printf '%s\n' 0
}
export -f nvidia-smi
export CUDA_VISIBLE_DEVICES=GPU-host-uuid
normalize_cuda_visibility
printf 'RESULT=%s\n' "$CUDA_VISIBLE_DEVICES"
'''
        )

        self.assertIn("Using CUDA_VISIBLE_DEVICES=0", output)
        self.assertIn("RESULT=0", output)

    def test_multi_gpu_allocation_is_not_rewritten(self):
        output = self.run_shell(
            r'''
nvidia-smi() {
  printf '%s\n' 0 1
}
export -f nvidia-smi
export CUDA_VISIBLE_DEVICES=GPU-a,GPU-b
normalize_cuda_visibility
printf 'RESULT=%s\n' "$CUDA_VISIBLE_DEVICES"
'''
        )

        self.assertIn("Detected 2 GPUs", output)
        self.assertIn("RESULT=GPU-a,GPU-b", output)

    def test_normalization_can_be_disabled(self):
        output = self.run_shell(
            r'''
nvidia-smi() {
  printf '%s\n' 0
}
export -f nvidia-smi
export CUDA_NORMALIZE_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=GPU-host-uuid
normalize_cuda_visibility
printf 'RESULT=%s\n' "$CUDA_VISIBLE_DEVICES"
'''
        )

        self.assertIn("normalization disabled", output)
        self.assertIn("RESULT=GPU-host-uuid", output)


if __name__ == "__main__":
    unittest.main()
