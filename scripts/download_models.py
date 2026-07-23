#!/usr/bin/env python3
"""Resumable, checksum-verified model provisioning for the RunPod volume."""

import argparse
import concurrent.futures
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
from urllib.error import HTTPError
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse


USER_AGENT = "grawthings-wan22-runpod/2"


def expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


def has_unresolved_template(value):
    if not isinstance(value, str):
        return False
    return bool(
        re.search(
            r"\{\{.+?\}\}|\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*",
            value,
        )
    )


def missing_required_env(names):
    missing = []
    for name in names or []:
        value = os.environ.get(str(name), "").strip()
        if not value or has_unresolved_template(value):
            missing.append(str(name))
    return missing


def cleaned_headers(raw):
    headers = {"User-Agent": USER_AGENT}
    for key, value in expand(raw or {}).items():
        value = str(value).strip()
        if value and not has_unresolved_template(value):
            headers[str(key)] = value
    return headers


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def marker_path(output):
    return output.with_name(f".{output.name}.verified.json")


def write_marker(output, expected_sha, expected_size):
    marker_path(output).write_text(
        json.dumps(
            {
                "sha256": expected_sha.lower(),
                "size_bytes": expected_size,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def valid_existing(output, expected_sha, expected_size, min_bytes):
    if not output.is_file():
        return False
    actual_size = output.stat().st_size
    if expected_size and actual_size != expected_size:
        return False
    if min_bytes and actual_size < min_bytes:
        return False

    marker = marker_path(output)
    if marker.is_file() and expected_sha:
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            if (
                str(data.get("sha256", "")).lower() == expected_sha.lower()
                and int(data.get("size_bytes") or actual_size) == actual_size
            ):
                return True
        except (OSError, ValueError, TypeError):
            pass

    if expected_sha:
        print(f"VERIFY existing: {output.name}")
        actual_sha = sha256_file(output)
        if actual_sha != expected_sha.lower():
            return False
        write_marker(output, expected_sha, actual_size)
    return True


def resolve_download_url(url, headers, timeout=90):
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except HTTPError as exc:
        # Authentication query parameters must never be echoed into Pod logs.
        raise RuntimeError(
            f"HTTP {exc.code} while resolving {redact_url(url)}"
        ) from None
    try:
        return response.geturl()
    finally:
        response.close()


def add_auth_query(url, env_name, parameter="token"):
    """Add a secret query token for download endpoints that require it.

    CivitAI accepts bearer headers for most API calls, but some protected model
    downloads require the token query parameter before redirecting to storage.
    Only the resolved, short-lived storage URL is handed to aria2/curl.
    """
    if not env_name:
        return url
    secret = os.environ.get(str(env_name), "").strip()
    if not secret or has_unresolved_template(secret):
        return url
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query[str(parameter or "token")] = secret
    return urlunparse(parsed._replace(query=urlencode(query)))


def redact_url(url):
    parsed = urlparse(url)
    sensitive = {"token", "api_key", "apikey", "key", "authorization"}
    query = [
        (key, "REDACTED" if key.lower() in sensitive else value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    return urlunparse(parsed._replace(query=urlencode(query)))


def run_aria2(url, part, connections, splits):
    part.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aria2c",
        "-x",
        str(connections),
        "-s",
        str(splits),
        "-k",
        "16M",
        "--continue=true",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--max-tries=10",
        "--retry-wait=3",
        "--connect-timeout=30",
        "--timeout=60",
        "--disk-cache=64M",
        "--summary-interval=10",
        "--console-log-level=warn",
        "--download-result=hide",
        "--user-agent",
        USER_AGENT,
        "-d",
        str(part.parent),
        "-o",
        part.name,
        url,
    ]
    subprocess.run(cmd, check=True)


def run_curl(url, part):
    part.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-fL",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "-A",
        USER_AGENT,
        "-o",
        str(part),
        url,
    ]
    subprocess.run(cmd, check=True)


def run_urllib(url, part):
    part.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)


def parse_huggingface_url(url):
    """Return (repo_id, revision, filename) for a Hub resolve URL."""
    parsed = urlparse(url)
    if parsed.hostname not in {"huggingface.co", "www.huggingface.co"}:
        return None
    repository, separator, remainder = parsed.path.lstrip("/").partition("/resolve/")
    revision, filename_separator, filename = remainder.partition("/")
    if (
        not separator
        or not filename_separator
        or len(repository.split("/")) != 2
        or not revision
        or not filename
    ):
        return None
    return unquote(repository), unquote(revision), unquote(filename)


def materialize_cached_file(cached_path, part):
    """Make the Hub cache blob the output without a second 10+ GB copy."""
    source = pathlib.Path(cached_path).resolve()
    part.parent.mkdir(parents=True, exist_ok=True)
    part.unlink(missing_ok=True)
    try:
        os.link(source, part)
        print(f"ZERO-COPY from HF cache: {part.name}")
    except OSError:
        # Different filesystems cannot hard-link. copyfileobj uses a large
        # buffer and keeps the portable fallback efficient.
        with source.open("rb") as src, part.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)


def run_hf_xet(url, part):
    coordinates = parse_huggingface_url(url)
    if not coordinates:
        raise ValueError(f"not a Hugging Face resolve URL: {url}")
    repo_id, revision, filename = coordinates
    from huggingface_hub import hf_hub_download

    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=os.environ.get("HF_TOKEN") or None,
    )
    materialize_cached_file(cached_path, part)


def extract_archive(archive, destination, selector):
    selector = str(selector).strip().lower()
    extensions = None
    if selector not in {"", "1", "true", "all", "*"}:
        extensions = {
            f".{item.strip().lstrip('.').lower()}"
            for item in selector.split(",")
            if item.strip()
        }
    destination.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            basename = os.path.basename(member)
            if not basename or member.endswith("/"):
                continue
            if extensions and pathlib.Path(basename).suffix.lower() not in extensions:
                continue
            target = destination / basename
            with bundle.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
            extracted.append(target)
    if not extracted:
        raise RuntimeError(f"no matching files in {archive.name}")
    archive.unlink()
    return extracted


def extracted_ready(entry, root):
    provided = [root / item for item in entry.get("provides", [])]
    sentinel = root / entry["path"]
    sentinel = sentinel.parent / f".{sentinel.name}.extracted.json"
    return bool(provided) and sentinel.is_file() and all(path.is_file() for path in provided)


def download_snapshot(entry, root, dry_run, max_workers):
    output = root / entry["path"]
    sentinel = output / ".snapshot-complete.json"
    name = entry.get("name") or entry["repo_id"]
    if sentinel.is_file() and any(path.is_file() for path in output.rglob("*")):
        print(f"OK snapshot: {name}")
        return
    if dry_run:
        print(f"WOULD DOWNLOAD snapshot: {name} -> {entry['path']}")
        return

    from huggingface_hub import snapshot_download

    print(f"DOWNLOAD snapshot: {name}")
    output.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    snapshot_download(
        repo_id=entry["repo_id"],
        revision=entry.get("revision"),
        local_dir=str(output),
        ignore_patterns=entry.get("ignore_patterns"),
        token=token,
        max_workers=max_workers,
    )
    sentinel.write_text(
        json.dumps(
            {"repo_id": entry["repo_id"], "revision": entry.get("revision")},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def download_file(
    entry,
    root,
    use_aria2,
    connections,
    splits,
    dry_run,
    prefer_hf_xet=True,
):
    entry = expand(entry)
    name = entry.get("name") or entry["path"]
    output = root / entry["path"]
    expected_sha = str(entry.get("sha256") or "").lower()
    expected_size = int(entry.get("size_bytes") or 0)
    min_bytes = int(entry.get("min_bytes") or expected_size or 1)
    required = entry.get("required", True)
    missing_env = missing_required_env(entry.get("requires_env"))
    if missing_env:
        message = f"missing required environment variable(s) for {name}: {', '.join(missing_env)}"
        if required:
            raise RuntimeError(message)
        print(f"WARN {message}", file=sys.stderr)
        return

    if entry.get("extract") and extracted_ready(entry, root):
        print(f"OK extracted: {name}")
        return
    if not entry.get("extract") and valid_existing(
        output, expected_sha, expected_size, min_bytes
    ):
        print(f"OK existing: {name}")
        return
    if dry_run:
        print(f"WOULD DOWNLOAD: {name} -> {entry['path']}")
        return

    if output.exists():
        print(f"REMOVE invalid/incomplete final file: {output.name}", file=sys.stderr)
        output.unlink()
    verified = marker_path(output)
    if verified.exists():
        verified.unlink()

    headers = cleaned_headers(entry.get("headers"))
    url = entry["url"]
    if has_unresolved_template(url):
        raise RuntimeError(f"unresolved environment template in URL for {name}")
    request_url = add_auth_query(
        url,
        entry.get("auth_query_env"),
        entry.get("auth_query_name", "token"),
    )

    part = output.with_name(f"{output.name}.part")
    try:
        print(f"DOWNLOAD: {name}")
        # Preserve old aria2 partials across image upgrades. New Hub downloads
        # use hf_xet, whose Rust backend adapts concurrency to the link.
        use_existing_partial = part.is_file() and part.stat().st_size > 0
        hub_coordinates = parse_huggingface_url(url)
        if prefer_hf_xet and hub_coordinates and not use_existing_partial:
            try:
                print(f"HF_XET: {name}")
                run_hf_xet(url, part)
            except Exception as exc:
                print(
                    f"WARN hf_xet failed for {name}; falling back to aria2: {exc}",
                    file=sys.stderr,
                )
                final_url = resolve_download_url(request_url, headers)
                if use_aria2 and shutil.which("aria2c"):
                    run_aria2(final_url, part, connections, splits)
                elif shutil.which("curl"):
                    run_curl(final_url, part)
                else:
                    run_urllib(final_url, part)
        elif use_aria2 and shutil.which("aria2c"):
            final_url = resolve_download_url(request_url, headers)
            run_aria2(final_url, part, connections, splits)
        elif shutil.which("curl"):
            final_url = resolve_download_url(request_url, headers)
            run_curl(final_url, part)
        else:
            final_url = resolve_download_url(request_url, headers)
            run_urllib(final_url, part)

        actual_size = part.stat().st_size
        if expected_size and actual_size != expected_size:
            raise RuntimeError(
                f"size mismatch for {name}: expected {expected_size}, got {actual_size}"
            )
        if actual_size < min_bytes:
            raise RuntimeError(f"downloaded file is too small for {name}: {actual_size}")
        if expected_sha:
            actual_sha = sha256_file(part)
            if actual_sha != expected_sha:
                part.unlink(missing_ok=True)
                raise RuntimeError(
                    f"SHA256 mismatch for {name}: expected {expected_sha}, got {actual_sha}"
                )

        part.replace(output)
        if entry.get("extract"):
            extracted = extract_archive(output, output.parent, entry["extract"])
            sentinel = output.parent / f".{output.name}.extracted.json"
            sentinel.write_text(
                json.dumps([path.name for path in extracted], sort_keys=True),
                encoding="utf-8",
            )
            print(f"EXTRACTED: {', '.join(path.name for path in extracted)}")
        else:
            write_marker(output, expected_sha, actual_size)
    except Exception:
        # Keep a partial aria2 download so a pod restart can resume it. A bad
        # checksum is explicitly deleted above.
        raise


def selected_groups(manifest, profile):
    profiles = manifest.get("profiles") or {}
    if profile not in profiles:
        choices = ", ".join(sorted(profiles))
        raise ValueError(f"unknown profile '{profile}' (choose one of: {choices})")
    return set(profiles[profile].get("include_groups") or [])


def largest_first(entries):
    return sorted(
        entries,
        key=lambda item: (int(item.get("size_bytes") or 0), item.get("name") or ""),
        reverse=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--profile", default=os.environ.get("MODEL_PROFILE", ""))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-aria2", action="store_true")
    parser.add_argument(
        "--connections", type=int, default=int(os.environ.get("ARIA2_CONNECTIONS", "8"))
    )
    parser.add_argument(
        "--splits", type=int, default=int(os.environ.get("ARIA2_SPLITS", "8"))
    )
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("DOWNLOAD_WORKERS", "4"))
    )
    parser.add_argument(
        "--hf-snapshot-workers",
        type=int,
        default=int(os.environ.get("HF_SNAPSHOT_WORKERS", "8")),
    )
    parser.add_argument("--no-hf-xet", action="store_true")
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = args.profile or manifest.get("default_profile") or "full"
    groups = selected_groups(manifest, profile)
    root = pathlib.Path(args.root)
    entries = [
        entry
        for entry in manifest.get("models", [])
        if entry.get("enabled", True) and entry.get("group") in groups
    ]
    print(f"MODEL PROFILE: {profile} ({len(entries)} assets)")

    # Fail before downloading tens of gigabytes when a protected source cannot
    # be accessed. This is especially useful for the CivitAI-hosted T2V v4 pair.
    env_errors = []
    for entry in entries:
        missing = missing_required_env(entry.get("requires_env"))
        if missing and entry.get("required", True):
            env_errors.append(
                f"{entry.get('name') or entry.get('path')}: {', '.join(missing)}"
            )
    if env_errors:
        raise RuntimeError(
            "profile prerequisites are missing:\n  - " + "\n  - ".join(env_errors)
        )

    ordered_entries = largest_first(entries)

    def provision(entry):
        if entry.get("repo_id"):
            download_snapshot(entry, root, args.dry_run, args.hf_snapshot_workers)
        else:
            download_file(
                entry,
                root,
                not args.no_aria2,
                args.connections,
                args.splits,
                args.dry_run,
                not args.no_hf_xet,
            )

    if args.dry_run:
        for entry in ordered_entries:
            provision(entry)
        return

    worker_count = max(1, min(args.workers, len(ordered_entries)))
    print(
        f"TRANSFER ENGINE: {worker_count} files in parallel; "
        f"aria2={args.connections} connections/file; "
        f"hf_xet={'off' if args.no_hf_xet else 'adaptive'}"
    )
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(provision, entry): entry for entry in ordered_entries}
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            try:
                future.result()
            except Exception as exc:
                name = entry.get("name") or entry.get("path")
                if entry.get("required", True):
                    failures.append(f"{name}: {exc}")
                    print(f"ERROR required asset failed: {name}: {exc}", file=sys.stderr)
                else:
                    print(f"WARN optional asset failed: {name}: {exc}", file=sys.stderr)

    if failures:
        raise RuntimeError("required asset download failures:\n  - " + "\n  - ".join(failures))


if __name__ == "__main__":
    main()
