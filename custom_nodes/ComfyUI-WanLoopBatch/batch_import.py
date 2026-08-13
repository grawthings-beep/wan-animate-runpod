"""Safe server-side import for a ten-image WAN loop ZIP."""

import pathlib
import re
import shutil
import uuid
import zipfile

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
PROMPT_FILENAMES = {"prompts.txt", "prompt.txt", "video_prompts.txt"}
MAX_ARCHIVE_BYTES = 300 * 1024 * 1024
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_TOTAL_BYTES = 500 * 1024 * 1024


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value).replace("\\", "/"))
    ]


def _is_symlink(info):
    return ((info.external_attr >> 16) & 0o170000) == 0o120000


def _safe_basename(value):
    name = pathlib.PurePosixPath(str(value).replace("\\", "/")).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", pathlib.Path(name).stem).strip(".-")
    return (stem or "image")[:80]


def _has_image_signature(header, suffix):
    suffix = suffix.lower()
    if suffix == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8\xff")
    if suffix == ".webp":
        return header.startswith(b"RIFF") and header[8:12] == b"WEBP"
    if suffix == ".bmp":
        return header.startswith(b"BM")
    if suffix == ".gif":
        return header.startswith((b"GIF87a", b"GIF89a"))
    return False


def _validated_members(bundle):
    members = []
    for info in bundle.infolist():
        if info.is_dir() or info.filename.startswith("__MACOSX/"):
            continue
        if info.flag_bits & 0x1:
            raise ValueError("encrypted ZIP entries are not supported")
        if _is_symlink(info):
            raise ValueError("ZIP symlinks are not allowed")
        suffix = pathlib.PurePosixPath(info.filename.replace("\\", "/")).suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            if info.file_size <= 0 or info.file_size > MAX_IMAGE_BYTES:
                raise ValueError(f"invalid image size in ZIP: {info.filename}")
            members.append(info)

    members.sort(key=lambda item: natural_key(item.filename))
    if len(members) != 10:
        raise ValueError(f"ZIP must contain exactly 10 images; found {len(members)}")
    if sum(item.file_size for item in members) > MAX_TOTAL_BYTES:
        raise ValueError("ZIP image payload is too large")
    return members


def extract_image_zip(archive_path, input_directory):
    """Flatten exactly ten verified image members into a unique input folder."""
    archive_path = pathlib.Path(archive_path)
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("ZIP is larger than the 300 MiB import limit")

    input_directory = pathlib.Path(input_directory).resolve()
    import_id = uuid.uuid4().hex
    relative_directory = pathlib.PurePosixPath("wan-loop-imports") / import_id
    final_directory = input_directory.joinpath(*relative_directory.parts)
    temporary_directory = final_directory.with_name(f".{import_id}.tmp")
    temporary_directory.mkdir(parents=True, exist_ok=False)

    results = []
    prompts_text = None
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            members = _validated_members(bundle)
            prompt_members = [
                item
                for item in bundle.infolist()
                if not item.is_dir()
                and pathlib.PurePosixPath(
                    item.filename.replace("\\", "/")
                ).name.casefold()
                in PROMPT_FILENAMES
            ]
            if len(prompt_members) > 1:
                raise ValueError("ZIP contains more than one recognized prompts.txt")
            if prompt_members:
                prompt_info = prompt_members[0]
                if prompt_info.file_size > 1024 * 1024:
                    raise ValueError("prompts.txt is larger than 1 MiB")
                prompts_text = bundle.read(prompt_info).decode("utf-8-sig")

            for slot, info in enumerate(members, start=1):
                suffix = pathlib.PurePosixPath(
                    info.filename.replace("\\", "/")
                ).suffix.lower()
                output_name = f"slot-{slot:02d}-{_safe_basename(info.filename)}{suffix}"
                output_path = temporary_directory / output_name
                with bundle.open(info) as source:
                    header = source.read(16)
                    if not _has_image_signature(header, suffix):
                        raise ValueError(f"invalid image payload: {info.filename}")
                    with output_path.open("wb") as target:
                        target.write(header)
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                try:
                    with Image.open(output_path) as image:
                        width, height = image.size
                        if width <= 0 or height <= 0 or width * height > 100_000_000:
                            raise ValueError(f"unsafe image dimensions: {info.filename}")
                        image.verify()
                except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
                    raise ValueError(f"invalid image payload: {info.filename}") from exc
                results.append(
                    {
                        "slot": slot,
                        "name": (relative_directory / output_name).as_posix(),
                        "display_name": info.filename.replace("\\", "/"),
                    }
                )

        final_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary_directory.replace(final_directory)
        return {"images": results, "prompts_text": prompts_text}
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
