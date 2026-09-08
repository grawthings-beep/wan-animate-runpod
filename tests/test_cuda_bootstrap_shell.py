import pathlib
import shutil
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


@unittest.skipUnless(BASH, "bash is required for startup shell tests")
class CudaBootstrapShellTests(unittest.TestCase):
    def test_visibility_is_never_rewritten_even_for_legacy_templates(self):
        source = (ROOT / "scripts/start.sh").read_text(encoding="utf-8")
        preamble = source[source.index("# Respect RunPod/user visibility"):source.index('WORKSPACE_DIR=')]
        for assignment in ("unset CUDA_VISIBLE_DEVICES", "export CUDA_VISIBLE_DEVICES=GPU-a",
                           "export CUDA_VISIBLE_DEVICES=0", "export CUDA_VISIBLE_DEVICES=-1",
                           "export CUDA_VISIBLE_DEVICES=''", "export CUDA_VISIBLE_DEVICES=GPU-a,GPU-b"):
            with self.subTest(assignment=assignment):
                script = "source scripts/common.sh\n" + assignment + r'''
export CUDA_NORMALIZE_VISIBLE_DEVICES=1
before="${CUDA_VISIBLE_DEVICES-<unset>}"
''' + preamble + r'''
after="${CUDA_VISIBLE_DEVICES-<unset>}"
test "$before" = "$after"
echo PRESERVED
'''
                result = subprocess.run([BASH, "-c", script], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("PRESERVED", result.stdout)


if __name__ == "__main__":
    unittest.main()
