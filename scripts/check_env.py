#!/usr/bin/env python3
"""Check CUDA visibility and every asset selected by a model profile."""

import argparse
import json
import pathlib
import sys


def selected_entries(manifest, profile):
    profiles = manifest.get("profiles") or {}
    if profile not in profiles:
        raise ValueError(f"unknown profile: {profile}")
    groups = set(profiles[profile].get("include_groups") or [])
    return [
        entry
        for entry in manifest.get("models", [])
        if entry.get("enabled", True) and entry.get("group") in groups
    ]


def entry_paths(root, entry):
    if entry.get("repo_id"):
        return [root / entry["path"] / ".snapshot-complete.json"]
    if entry.get("extract"):
        return [root / item for item in entry.get("provides", [])]
    return [root / entry["path"]]


def missing_by_requirement(root, entries):
    required_missing = []
    optional_missing = []
    for entry in entries:
        absent = [path for path in entry_paths(root, entry) if not path.exists()]
        target = required_missing if entry.get("required", True) else optional_missing
        target.extend(absent)
    return required_missing, optional_missing


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model-root", default="/workspace/comfyui")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    try:
        import torch

        print(
            f"[check_env] torch={torch.__version__} "
            f"cuda_available={torch.cuda.is_available()}"
        )
        if torch.cuda.is_available():
            print(f"[check_env] device={torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[check_env] torch import failed: {exc}")

    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    root = pathlib.Path(args.model_root)
    entries = selected_entries(manifest, args.profile)
    required_missing, optional_missing = missing_by_requirement(root, entries)
    for entry in entries:
        paths = entry_paths(root, entry)
        absent = [path for path in paths if not path.exists()]
        if absent:
            label = "MISSING" if entry.get("required", True) else "OPTIONAL MISSING"
            print(f"[check_env] {label} {entry['name']}: {', '.join(map(str, absent))}")
        else:
            print(f"[check_env] OK {entry['name']}")

    print(
        f"[check_env] profile={args.profile} assets={len(entries)} "
        f"required_missing={len(required_missing)} "
        f"optional_missing={len(optional_missing)}"
    )
    if required_missing and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
