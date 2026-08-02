"""Pure filesystem helpers for the WAN loop queue custom nodes."""

import json
import os
import pathlib
import re
import zipfile


BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
VIDEO_SUFFIXES = {".gif", ".mkv", ".mov", ".mp4", ".webm", ".webp"}


def validate_batch_id(batch_id):
    value = str(batch_id or "").strip()
    if not BATCH_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "batch_id must be 1-80 characters containing only letters, "
            "numbers, underscores, and hyphens"
        )
    return value


def _resolved_within(root, candidate):
    root = pathlib.Path(root).resolve()
    candidate = pathlib.Path(candidate).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output path escapes batch directory: {candidate}") from exc
    return candidate


def video_from_vhs_filenames(filenames):
    if not isinstance(filenames, (list, tuple)) or len(filenames) != 2:
        raise ValueError("VHS filenames payload is malformed")
    save_output, output_files = filenames
    if not save_output:
        raise ValueError("VHS Video Combine must have save_output enabled")
    if not isinstance(output_files, (list, tuple)):
        raise ValueError("VHS filenames payload does not contain output files")

    videos = [
        pathlib.Path(path)
        for path in output_files
        if pathlib.Path(path).suffix.lower() in VIDEO_SUFFIXES
    ]
    if not videos:
        raise ValueError("VHS Video Combine did not return a video file")
    video = videos[-1].resolve()
    if not video.is_file():
        raise FileNotFoundError(f"generated video does not exist: {video}")
    return video


def _write_json_atomic(path, payload):
    path = pathlib.Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def record_video_and_maybe_archive(
    output_root,
    batch_id,
    slot,
    image_name,
    positive_prompt,
    filenames,
    expected_count=10,
):
    """Record one completed slot and return a ZIP path after the final slot."""
    batch_id = validate_batch_id(batch_id)
    slot = int(slot)
    expected_count = int(expected_count)
    if expected_count < 1 or expected_count > 100:
        raise ValueError("expected_count must be between 1 and 100")
    if slot < 1 or slot > expected_count:
        raise ValueError(f"slot must be between 1 and {expected_count}")

    output_root = pathlib.Path(output_root).resolve()
    batch_dir = output_root / "Video" / "loop-batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    video = _resolved_within(batch_dir, video_from_vhs_filenames(filenames))

    manifest_path = batch_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("batch_id") != batch_id:
            raise ValueError("existing manifest belongs to a different batch")
        if int(manifest.get("expected_count", 0)) != expected_count:
            raise ValueError("existing manifest has a different expected_count")
    else:
        manifest = {
            "batch_id": batch_id,
            "expected_count": expected_count,
            "videos": {},
        }

    relative_video = video.relative_to(batch_dir).as_posix()
    manifest["videos"][str(slot)] = {
        "slot": slot,
        "image": str(image_name),
        "positive_prompt": str(positive_prompt),
        "server_file": relative_video,
    }
    _write_json_atomic(manifest_path, manifest)

    completed = len(manifest["videos"])
    if slot != expected_count:
        return {
            "archive": None,
            "batch_dir": batch_dir,
            "completed": completed,
            "expected": expected_count,
        }

    missing = [
        number
        for number in range(1, expected_count + 1)
        if str(number) not in manifest["videos"]
    ]
    if missing:
        raise RuntimeError(
            "final slot completed but earlier slots are missing: "
            + ", ".join(map(str, missing))
        )

    archive = batch_dir / f"{batch_id}.zip"
    temporary_archive = archive.with_suffix(".zip.tmp")
    archive_manifest = json.loads(json.dumps(manifest))
    try:
        with zipfile.ZipFile(
            temporary_archive, "w", compression=zipfile.ZIP_STORED
        ) as bundle:
            for number in range(1, expected_count + 1):
                entry = archive_manifest["videos"][str(number)]
                source = _resolved_within(
                    batch_dir, batch_dir / entry["server_file"]
                )
                if not source.is_file():
                    raise FileNotFoundError(
                        f"completed slot video disappeared: {source}"
                    )
                archive_name = f"slot-{number:02d}{source.suffix.lower()}"
                bundle.write(source, archive_name)
                entry["archive_file"] = archive_name
            bundle.writestr(
                "manifest.json",
                json.dumps(archive_manifest, ensure_ascii=False, indent=2) + "\n",
            )
    except Exception:
        temporary_archive.unlink(missing_ok=True)
        raise
    os.replace(temporary_archive, archive)
    return {
        "archive": archive,
        "batch_dir": batch_dir,
        "completed": completed,
        "expected": expected_count,
    }
