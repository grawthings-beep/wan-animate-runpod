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
            self.module._expanded_box((5, 10, 25, 30), 30, 35, 1.0),
            (0, 0, 30, 35),
        )

    def test_temporal_gap_fill_preserves_detected_contours_and_wraps_seam(self):
        first = np.zeros((3, 3), dtype=np.bool_)
        third = np.zeros((3, 3), dtype=np.bool_)
        first[1, 0] = True
        third[1, 2] = True
        filled = self.module._fill_short_circular_gaps(
            [first, None, third, None], 1
        )
        self.assertTrue(np.array_equal(filled[0], first))
        self.assertTrue(np.array_equal(filled[2], third))
        self.assertIsNotNone(filled[1])
        self.assertIsNotNone(filled[3])

    def test_fixed_grid_mosaic_uses_stationary_blocks(self):
        image = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
        result = self.module._fixed_grid_mosaic(image, 3)
        self.assertTrue(np.all(result[0:3, 0:3] == result[0, 0]))
        self.assertTrue(np.all(result[3:6, 3:6] == result[3, 3]))
        self.assertFalse(np.array_equal(result[0, 0], result[3, 3]))

    def test_default_targets_exclude_anus_and_large_context_classes(self):
        ids = self.module._selected_class_ids(
            {
                0: "nipples",
                1: "pussy",
                2: "anus",
                3: "penis",
                4: "cross-section",
                5: "x-ray",
                6: "testicles",
            },
            self.module.DEFAULT_CLASSES,
        )
        self.assertEqual(ids, [1, 3, 6])

    def test_auto_block_size_matches_iphone_short_side_rule(self):
        self.assertEqual(self.module._resolve_block_size(0, 528, 704), 11)
        self.assertEqual(self.module._resolve_block_size(36, 528, 704), 36)


if __name__ == "__main__":
    unittest.main()
