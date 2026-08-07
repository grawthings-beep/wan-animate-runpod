import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "custom_nodes"
    / "ComfyUI-WanLoopBatch"
    / "mosaic_nodes.py"
)


def load_module():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = "/workspace/comfyui/models"
    sys.modules["folder_paths"] = folder_paths
    spec = importlib.util.spec_from_file_location("wan_mosaic_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AutoMosaicHelperTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_box_expansion_is_clamped_to_frame(self):
        self.assertEqual(
            self.module._expanded_box((5, 10, 25, 30), 30, 35, 100),
            (0, 0, 30, 35),
        )

    def test_temporal_union_wraps_across_loop_seam(self):
        masks = np.zeros((4, 2, 2), dtype=np.bool_)
        masks[0, 0, 0] = True
        smoothed = self.module._circular_temporal_union(masks, 1)
        self.assertTrue(smoothed[0, 0, 0])
        self.assertTrue(smoothed[1, 0, 0])
        self.assertTrue(smoothed[3, 0, 0])
        self.assertFalse(smoothed[2, 0, 0])

    def test_fixed_grid_mosaic_uses_stationary_blocks(self):
        image = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
        result = self.module._fixed_grid_mosaic(image, 3)
        self.assertTrue(np.all(result[0:3, 0:3] == result[0, 0]))
        self.assertTrue(np.all(result[3:6, 3:6] == result[3, 3]))
        self.assertFalse(np.array_equal(result[0, 0], result[3, 3]))

    def test_default_targets_exclude_large_context_class(self):
        ids = self.module._selected_class_ids(
            {0: "anus", 1: "make_love", 2: "nipple", 3: "penis", 4: "vagina"},
            self.module.DEFAULT_CLASSES,
            False,
        )
        self.assertEqual(ids, [0, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
