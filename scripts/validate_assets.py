#!/usr/bin/env python3
"""Static integrity checks for workflows, model manifests, and custom nodes."""

import json
import pathlib
import re
import sys
from urllib.parse import urlparse


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "wan22-models.json"
DEPENDENCIES_PATH = ROOT / "config" / "workflow-dependencies.json"
CUSTOM_NODES_PATH = ROOT / "custom_nodes.txt"
LOOP_CUSTOM_NODES_PATH = ROOT / "custom_nodes.loop.txt"
WORKFLOW_PATHS = sorted((ROOT / "workflows").glob("*_runpod.json"))
MODEL_EXTENSIONS = {".safetensors", ".gguf", ".pth", ".pkl"}
UUID_TYPE = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", re.I)


def walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_strings(item)


def iter_nodes(workflow):
    yield from workflow.get("nodes", [])
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        yield from subgraph.get("nodes", [])


def read_custom_node_names(path, errors):
    names = set()
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != 3:
            errors.append(f"{path.name}:{line_number}: expected name|url|commit")
            continue
        name, url, commit = parts
        if name in names:
            errors.append(f"{path.name}:{line_number}: duplicate {name}")
        names.add(name)
        if urlparse(url).scheme != "https":
            errors.append(f"{path.name}:{line_number}: URL must use HTTPS")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            errors.append(f"{path.name}:{line_number}: commit is not pinned")
    return names


def validate_manifest(manifest, errors):
    profiles = manifest.get("profiles") or {}
    if manifest.get("default_profile") not in profiles:
        errors.append("manifest default_profile does not exist")

    paths = set()
    names = set()
    groups = set()
    provided_basenames = set()
    basename_groups = {}
    for index, entry in enumerate(manifest.get("models", [])):
        label = f"manifest models[{index}]"
        path = entry.get("path", "")
        name = entry.get("name", "")
        group = entry.get("group", "")
        if not path or pathlib.PurePosixPath(path).is_absolute() or ".." in pathlib.PurePosixPath(path).parts:
            errors.append(f"{label}: unsafe/missing relative path")
        if path in paths:
            errors.append(f"{label}: duplicate path {path}")
        paths.add(path)
        if name in names:
            errors.append(f"{label}: duplicate name {name}")
        names.add(name)
        groups.add(group)
        provided_basenames.add(pathlib.PurePosixPath(path).name)
        basename_groups[pathlib.PurePosixPath(path).name] = group
        for provided in entry.get("provides", []):
            provided_basenames.add(pathlib.PurePosixPath(provided).name)
            basename_groups[pathlib.PurePosixPath(provided).name] = group

        if entry.get("repo_id"):
            if not entry.get("revision"):
                errors.append(f"{label}: snapshot revision must be pinned")
        else:
            if urlparse(entry.get("url", "")).scheme != "https":
                errors.append(f"{label}: URL must use HTTPS")
            sha = entry.get("sha256", "")
            if not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
                errors.append(f"{label}: invalid SHA256")
            if int(entry.get("size_bytes") or 0) <= 0:
                errors.append(f"{label}: size_bytes must be positive")
        auth_query_env = entry.get("auth_query_env")
        if auth_query_env and auth_query_env not in (entry.get("requires_env") or []):
            errors.append(
                f"{label}: auth_query_env must also be listed in requires_env"
            )

    used_groups = set()
    for profile_name, profile in profiles.items():
        included = set(profile.get("include_groups") or [])
        unknown = included - groups
        if unknown:
            errors.append(f"profile {profile_name}: unknown groups {sorted(unknown)}")
        used_groups.update(included)
    if groups - used_groups:
        errors.append(f"manifest groups unused by every profile: {sorted(groups - used_groups)}")
    return provided_basenames, basename_groups


def validate_workflows(
    provided_basenames,
    basename_groups,
    installed_packs,
    dependencies,
    manifest_profiles,
    errors,
    workflow_paths=WORKFLOW_PATHS,
):
    core = set(dependencies.get("core_node_types") or [])
    custom = dependencies.get("custom_node_types") or {}
    if not workflow_paths:
        errors.append("no generated RunPod workflows found")
        return

    for path in workflow_paths:
        workflow = json.loads(path.read_text(encoding="utf-8"))
        bundle = workflow.get("extra", {}).get("runpod_bundle", {})
        profile = bundle.get("profile")
        if profile not in manifest_profiles:
            errors.append(
                f"{path.name}: unknown/missing runpod_bundle profile {profile!r}"
            )
        for node in iter_nodes(workflow):
            node_type = str(node.get("type", ""))
            if node_type in core or UUID_TYPE.fullmatch(node_type):
                continue
            pack = custom.get(node_type)
            if not pack:
                errors.append(f"{path.name}: unmapped node type {node_type!r}")
            elif pack not in installed_packs:
                errors.append(f"{path.name}: node {node_type!r} needs missing pack {pack}")

        referenced = set()
        for value in walk_strings(workflow):
            suffix = pathlib.PurePath(value).suffix.lower()
            if suffix in MODEL_EXTENSIONS:
                referenced.add(pathlib.PurePath(value).name)
        missing = referenced - provided_basenames
        if missing:
            errors.append(f"{path.name}: models absent from manifest: {sorted(missing)}")
        if bundle.get("requires_all_referenced_assets") and profile in manifest_profiles:
            profile_groups = set(
                manifest_profiles[profile].get("include_groups") or []
            )
            outside_profile = {
                name
                for name in referenced
                if basename_groups.get(name) not in profile_groups
            }
            if outside_profile:
                errors.append(
                    f"{path.name}: models absent from profile {profile}: "
                    f"{sorted(outside_profile)}"
                )
        raw = path.read_text(encoding="utf-8")
        if re.search(r"[A-Za-z]:\\\\", raw):
            errors.append(f"{path.name}: contains a stale Windows absolute path")


def main():
    errors = []
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dependencies = json.loads(DEPENDENCIES_PATH.read_text(encoding="utf-8"))
    installed_packs = read_custom_node_names(CUSTOM_NODES_PATH, errors)
    loop_installed_packs = read_custom_node_names(LOOP_CUSTOM_NODES_PATH, errors)
    if not loop_installed_packs.issubset(installed_packs):
        errors.append(
            "custom_nodes.loop.txt contains packs absent from custom_nodes.txt: "
            f"{sorted(loop_installed_packs - installed_packs)}"
        )
    bundled_packs = dependencies.get("bundled_custom_node_packs") or []
    for pack in bundled_packs:
        package = ROOT / "custom_nodes" / pack
        if not (package / "__init__.py").is_file():
            errors.append(f"bundled custom-node pack is missing: {pack}")
        installed_packs.add(pack)
        loop_installed_packs.add(pack)
    provided, basename_groups = validate_manifest(manifest, errors)
    validate_workflows(
        provided,
        basename_groups,
        installed_packs,
        dependencies,
        manifest.get("profiles") or {},
        errors,
    )
    validate_workflows(
        provided,
        basename_groups,
        loop_installed_packs,
        dependencies,
        manifest.get("profiles") or {},
        errors,
        workflow_paths=[
            path for path in WORKFLOW_PATHS if "seamless_loop" in path.name
        ],
    )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"OK: {len(WORKFLOW_PATHS)} workflows, "
        f"{len(manifest['models'])} assets, {len(installed_packs)} full packs, "
        f"{len(loop_installed_packs)} loop packs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
