#!/usr/bin/env python3
"""Install only two managed canvases; preserve retired/edited copies outside UI."""

import argparse
import copy
import hashlib
import json
import pathlib
import re
import shutil


CURRENT_NAMES = ("wan22_loop_single_runpod.json", "wan22_loop_batch10_runpod.json")
LEGACY_NAMES = {
    "wan22_smooth_v6_aio_runpod.json",
    "wan22_smooth_v6_i2v_auto_mosaic_runpod.json",
    "wan22_native_enhanced_lightning_longvideo_runpod.json",
    *(f"wan22_smooth_v6_seamless_loop{suffix}_runpod.json" for suffix in (
        "", "_core", "_batch10", "_batch10_core", "_auto_mosaic",
        "_core_auto_mosaic", "_batch10_auto_mosaic", "_batch10_core_auto_mosaic",
    )),
}


def is_managed_name(name):
    original = re.sub(r"-bundle-[0-9a-f]{12}(?=\.json$)", "", name)
    return original in LEGACY_NAMES or original in CURRENT_NAMES


def adapt_profile(graph, manifest, profile):
    """One canvas per job type, even when downloading the smaller core pack."""
    graph = copy.deepcopy(graph)
    groups = set(manifest["profiles"][profile]["include_groups"])
    available = {pathlib.PurePosixPath(item["path"]).name for item in manifest["models"]
                 if item["group"] in groups}
    for node in graph["nodes"]:
        if node["type"] == "Power Lora Loader (rgthree)":
            node["widgets_values"] = [item for item in node["widgets_values"]
                if not isinstance(item, dict) or not item.get("lora") or item["lora"] in available]
    graph["extra"]["runpod_bundle"]["profile"] = profile
    return graph


def _archive(path, backup):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Refusing to archive non-regular managed workflow: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    target = backup / f"{path.stem}-{digest}.json"
    if target.is_symlink():
        raise ValueError(f"Refusing symlink backup target: {target}")
    if target.exists():
        if target.read_bytes() != path.read_bytes():
            raise ValueError(f"Backup collision: {target}")
    else:
        shutil.copy2(path, target)
    path.unlink()  # A verified recoverable copy exists before removing from UI.
    print(f"Archived managed workflow: {path.name} -> {target}")


def install(source, destination, backup, manifest, profile):
    source, destination, backup = [pathlib.Path(p).resolve() for p in (source, destination, backup)]
    if backup == destination or backup.is_relative_to(destination) or destination.is_relative_to(backup):
        raise ValueError("Backup and workflow directories must be separate, non-nested paths.")
    payloads = {}
    for name in CURRENT_NAMES:
        graph = json.loads((source / name).read_text(encoding="utf-8"))
        graph = adapt_profile(graph, manifest, profile)
        payloads[name] = (json.dumps(graph, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    destination.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=True)
    # Validate all exact targets first. Arbitrary user workflows are untouched.
    managed = [p for p in destination.iterdir() if is_managed_name(p.name)]
    for path in managed:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Refusing non-regular managed workflow: {path}")
    for path in managed:
        expected = payloads.get(path.name)
        if expected is None or path.read_bytes() != expected:
            _archive(path, backup)
    for name, content in payloads.items():
        path = destination / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(content)
            print(f"Installed bundled workflow: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--destination", required=True, type=pathlib.Path)
    parser.add_argument("--backup", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    install(args.source, args.destination, args.backup,
            json.loads(args.manifest.read_text(encoding="utf-8")), args.profile)


if __name__ == "__main__":
    main()
