import os

import folder_paths
from nodes import LoadImage

from .batch_utils import record_video_and_maybe_archive, validate_batch_id


SLOT_TYPE = "WAN_LOOP_QUEUE_SLOT"
CONTEXT_TYPE = "WAN_LOOP_QUEUE_CONTEXT"


class WanLoopQueueSlot:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            name
            for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
        ]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "positive_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
            }
        }

    RETURN_TYPES = (SLOT_TYPE,)
    RETURN_NAMES = ("slot",)
    FUNCTION = "pack"
    CATEGORY = "WAN Loop Queue"

    @classmethod
    def VALIDATE_INPUTS(cls, image, positive_prompt):
        image_result = LoadImage.VALIDATE_INPUTS(image)
        if image_result is not True:
            return image_result
        if not str(positive_prompt).strip():
            return "positive_prompt cannot be empty"
        return True

    @classmethod
    def IS_CHANGED(cls, image, positive_prompt):
        return f"{LoadImage.IS_CHANGED(image)}:{positive_prompt}"

    def pack(self, image, positive_prompt):
        return ({"image": image, "positive_prompt": positive_prompt.strip()},)


class WanLoopQueueSelector:
    @classmethod
    def INPUT_TYPES(cls):
        slots = {f"slot_{number:02d}": (SLOT_TYPE,) for number in range(1, 11)}
        return {
            "required": {
                "active_slot": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "control_after_generate": "increment",
                    },
                ),
                "batch_id": (
                    "STRING",
                    {"default": "click-queue-10-button", "multiline": False},
                ),
                **slots,
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", CONTEXT_TYPE)
    RETURN_NAMES = ("image", "positive_prompt", "filename_prefix", "context")
    FUNCTION = "select"
    CATEGORY = "WAN Loop Queue"

    @classmethod
    def IS_CHANGED(cls, active_slot, batch_id, **_kwargs):
        return f"{batch_id}:{int(active_slot)}"

    def select(self, active_slot, batch_id, **kwargs):
        slot_number = int(active_slot)
        if slot_number < 1 or slot_number > 10:
            raise ValueError("active_slot must be between 1 and 10")
        batch_id = validate_batch_id(batch_id)
        slot = kwargs[f"slot_{slot_number:02d}"]
        image_name = str(slot["image"])
        positive_prompt = str(slot["positive_prompt"]).strip()
        if not positive_prompt:
            raise ValueError(f"slot {slot_number:02d} positive prompt is empty")

        image, _mask = LoadImage().load_image(image_name)
        filename_prefix = (
            f"Video/loop-batches/{batch_id}/slot-{slot_number:02d}"
        )
        context = {
            "batch_id": batch_id,
            "slot": slot_number,
            "image": image_name,
            "positive_prompt": positive_prompt,
        }
        return image, positive_prompt, filename_prefix, context


class WanLoopBatchFinalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filenames": ("VHS_FILENAMES",),
                "context": (CONTEXT_TYPE,),
                "expected_count": (
                    "INT",
                    {"default": 10, "min": 1, "max": 100, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "finalize"
    OUTPUT_NODE = True
    CATEGORY = "WAN Loop Queue"

    def finalize(self, filenames, context, expected_count):
        result = record_video_and_maybe_archive(
            folder_paths.get_output_directory(),
            context["batch_id"],
            context["slot"],
            context["image"],
            context["positive_prompt"],
            filenames,
            expected_count,
        )
        status = (
            f"batch={context['batch_id']} slot={context['slot']:02d} "
            f"completed={result['completed']}/{result['expected']}"
        )
        ui = {"wan_loop_batch_progress": [status]}
        archive = result["archive"]
        if archive is not None:
            output_root = os.path.realpath(folder_paths.get_output_directory())
            relative = archive.relative_to(output_root)
            ui["wan_loop_batch_download"] = [
                {
                    "filename": relative.name,
                    "subfolder": relative.parent.as_posix(),
                    "type": "output",
                    "batch_id": context["batch_id"],
                }
            ]
            status += f" archive={relative.as_posix()}"
        return {"ui": ui, "result": (status,)}


NODE_CLASS_MAPPINGS = {
    "WanLoopQueueSlot": WanLoopQueueSlot,
    "WanLoopQueueSelector": WanLoopQueueSelector,
    "WanLoopBatchFinalize": WanLoopBatchFinalize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanLoopQueueSlot": "WAN Loop Image + Prompt Slot",
    "WanLoopQueueSelector": "WAN Queue 10 Loops Sequentially",
    "WanLoopBatchFinalize": "WAN ZIP + Auto Download After Slot 10",
}
