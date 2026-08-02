import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_nodes" / "ComfyUI-WanLoopBatch"


class FakeLoadImage:
    loaded = []

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        return True

    @classmethod
    def IS_CHANGED(cls, image):
        return f"hash:{image}"

    def load_image(self, image):
        self.loaded.append(image)
        return f"tensor:{image}", "mask"


def load_package(input_directory):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: str(input_directory)
    folder_paths.get_output_directory = lambda: str(input_directory)
    folder_paths.filter_files_content_types = lambda files, _types: files
    nodes = types.ModuleType("nodes")
    nodes.LoadImage = FakeLoadImage
    sys.modules["folder_paths"] = folder_paths
    sys.modules["nodes"] = nodes

    name = "wan_loop_batch_under_test"
    spec = importlib.util.spec_from_file_location(
        name,
        PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(PACKAGE_DIR)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package


class WanLoopNodeTests(unittest.TestCase):
    def test_package_imports_and_registers_all_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            package = load_package(pathlib.Path(directory))
            self.assertEqual(
                set(package.NODE_CLASS_MAPPINGS),
                {
                    "WanLoopQueueSlot",
                    "WanLoopQueueSelector",
                    "WanLoopBatchFinalize",
                },
            )
            self.assertEqual(package.WEB_DIRECTORY, "./web")

    def test_selector_loads_only_the_active_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            package = load_package(pathlib.Path(directory))
            selector = package.NODE_CLASS_MAPPINGS["WanLoopQueueSelector"]()
            slots = {
                f"slot_{number:02d}": {
                    "image": f"image-{number:02d}.png",
                    "positive_prompt": f"prompt {number:02d}",
                }
                for number in range(1, 11)
            }
            FakeLoadImage.loaded = []

            image, prompt, prefix, context = selector.select(
                7, "loop10-selector", **slots
            )

            self.assertEqual(FakeLoadImage.loaded, ["image-07.png"])
            self.assertEqual(image, "tensor:image-07.png")
            self.assertEqual(prompt, "prompt 07")
            self.assertEqual(
                prefix, "Video/loop-batches/loop10-selector/slot-07"
            )
            self.assertEqual(context["slot"], 7)


if __name__ == "__main__":
    unittest.main()
