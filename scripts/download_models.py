#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile


def extract_archive(archive, dest_dir, what):
    """Extract members of a zip into dest_dir (flattened to basenames).

    `what` selects which members to keep: a comma-separated extension list
    (e.g. "pt" or "pt,safetensors") or a truthy value ("all"/"*"/True) for
    everything. Subdirectories in the archive are flattened. The archive is
    removed afterwards. Returns the list of extracted filenames.
    """
    exts = None
    if isinstance(what, str) and what.strip().lower() not in ("", "1", "true", "all", "*"):
        exts = {"." + e.strip().lstrip(".").lower() for e in what.split(",") if e.strip()}
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            base = os.path.basename(member)
            if not base or member.endswith("/"):
                continue
            if exts and os.path.splitext(base)[1].lower() not in exts:
                continue
            target = dest_dir / base
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024 * 8)
            extracted.append(base)
    archive.unlink()
    return extracted


def expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def has_unresolved_template(value):
    if not isinstance(value, str):
        return False
    return bool(re.search(r"\{\{.+?\}\}|\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*", value))


def missing_required_env(names):
    missing = []
    for name in names or []:
        value = os.environ.get(str(name), "").strip()
        if not value or has_unresolved_template(value):
            missing.append(str(name))
    return missing


def cleaned_headers(raw):
    headers = {}
    for key, value in (raw or {}).items():
        value = expand(str(value)).strip()
        if not value:
            continue
        if key.lower() == "authorization" and value.lower() in {"bearer", "bearer ${hf_token}", "bearer ${civitai_token}"}:
            continue
        if key.lower() == "authorization" and value.lower().startswith("bearer ") and len(value.split(" ", 1)[1].strip()) == 0:
            continue
        headers[key] = value
    return headers


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 8), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_download_url(url, headers, timeout=90):
    request = urllib.request.Request(url, headers=headers | {"User-Agent": headers.get("User-Agent", "runpod-wan-animate-template")})
    response = urllib.request.urlopen(request, timeout=timeout)
    try:
        return response.geturl()
    finally:
        response.close()


def run_aria2(url, output, connections, splits):
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aria2c",
        "-x",
        str(connections),
        "-s",
        str(splits),
        "-k",
        "1M",
        "--continue=true",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--summary-interval=10",
        "--console-log-level=warn",
        "-d",
        str(output.parent),
        "-o",
        output.name,
        url,
    ]
    subprocess.run(cmd, check=True)


def run_curl(url, output, headers):
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    cmd = [
        "curl",
        "-fL",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--retry-all-errors",
        "-A",
        headers.get("User-Agent", "Mozilla/5.0"),
    ]
    for key, value in headers.items():
        if key.lower() == "user-agent":
            continue
        cmd.extend(["-H", f"{key}: {value}"])
    cmd.extend(["-o", str(tmp), url])
    subprocess.run(cmd, check=True)
    tmp.replace(output)


def run_urllib(url, output, headers):
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    request = urllib.request.Request(url, headers=headers | {"User-Agent": headers.get("User-Agent", "runpod-wan-animate-template")})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024 * 8)
    tmp.replace(output)


def download(entry, root, use_aria2, connections, splits):
    entry = expand(entry)
    name = entry.get("name") or entry.get("path")
    url = entry["url"]
    output = root / entry["path"]
    expected_sha = (entry.get("sha256") or "").upper()
    min_bytes = int(entry.get("min_bytes") or 0)
    required = entry.get("required", True)
    headers = cleaned_headers(entry.get("headers"))
    method = str(entry.get("method") or "").lower()
    extract = entry.get("extract")
    sentinel = output.parent / ("." + output.name + ".extracted")
    missing_env = missing_required_env(entry.get("requires_env"))
    if missing_env:
        message = f"missing required env for {name}: {', '.join(missing_env)}"
        if required:
            raise RuntimeError(message)
        print(f"WARN optional model skipped: {message}", file=sys.stderr)
        if output.exists() and min_bytes and output.stat().st_size < min_bytes:
            output.unlink()
        return

    if has_unresolved_template(url):
        message = f"unresolved template in url for {name}"
        if required:
            raise RuntimeError(message)
        print(f"WARN optional model skipped: {message}", file=sys.stderr)
        if output.exists() and min_bytes and output.stat().st_size < min_bytes:
            output.unlink()
        return

    if extract and sentinel.exists():
        print(f"OK extracted: {name}")
        return

    if output.exists() and output.stat().st_size > 0:
        if min_bytes and output.stat().st_size < min_bytes:
            print(f"Too small, redownloading: {name}", file=sys.stderr)
            output.unlink()
        elif expected_sha:
            actual_sha = sha256_file(output)
            if actual_sha == expected_sha:
                print(f"OK existing: {name}")
                return
            print(f"SHA mismatch, redownloading: {name}", file=sys.stderr)
            output.unlink()
        elif not extract:
            print(f"SKIP existing: {name}")
            return

    try:
        print(f"DOWNLOAD: {name}")
        if method == "curl" and shutil.which("curl"):
            run_curl(url, output, headers)
        elif use_aria2 and entry.get("use_aria2", True) and shutil.which("aria2c"):
            final_url = resolve_download_url(url, headers)
            run_aria2(final_url, output, connections, splits)
        else:
            run_urllib(url, output, headers)

        if min_bytes and output.stat().st_size < min_bytes:
            raise RuntimeError(f"downloaded file is too small for {output.name}: {output.stat().st_size} bytes")
        if expected_sha:
            actual_sha = sha256_file(output)
            if actual_sha != expected_sha:
                raise RuntimeError(f"sha256 mismatch for {output.name}: expected {expected_sha}, got {actual_sha}")
        if extract:
            names = extract_archive(output, output.parent, extract)
            if not names:
                raise RuntimeError(f"no matching files extracted from {output.name}")
            sentinel.write_text("\n".join(names))
            print(f"EXTRACTED ({name}): {', '.join(names)}")
    except Exception as exc:
        tmp = output.with_suffix(output.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        if output.exists():
            output.unlink()
        if required:
            raise
        print(f"WARN optional model failed: {name}: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--no-aria2", action="store_true")
    parser.add_argument("--connections", type=int, default=int(os.environ.get("ARIA2_CONNECTIONS", "16")))
    parser.add_argument("--splits", type=int, default=int(os.environ.get("ARIA2_SPLITS", "16")))
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    for entry in manifest.get("models", []):
        if not entry.get("enabled", True):
            print(f"SKIP disabled: {entry.get('name') or entry.get('path')}")
            continue
        download(entry, root, not args.no_aria2, args.connections, args.splits)


if __name__ == "__main__":
    main()
