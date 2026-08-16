#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TILE_LIST = SCRIPT_DIR / "woerden_tiles_v20250903.txt"
DEFAULT_CHECKSUM_FILE = SCRIPT_DIR / "woerden_archives_v20250903.sha256"
DEFAULT_RELEASE = "v20250903"
DEFAULT_BASE_URL = "https://data.3dbag.nl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the pinned 3DBAG OBJ tiles used by the HELIOS++ pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tile-list", type=Path, default=DEFAULT_TILE_LIST)
    parser.add_argument("--checksum-file", type=Path, default=DEFAULT_CHECKSUM_FILE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-zip-test", action="store_true")
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_tile_ids(path):
    tile_ids = []
    seen = set()
    for raw in path.expanduser().read_text(encoding="utf-8").splitlines():
        tile_id = raw.split("#", 1)[0].strip()
        if not tile_id:
            continue
        parts = tile_id.split("-")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError("Invalid 3DBAG tile ID: {}".format(tile_id))
        if tile_id in seen:
            raise ValueError("Duplicate 3DBAG tile ID: {}".format(tile_id))
        seen.add(tile_id)
        tile_ids.append(tile_id)
    if not tile_ids:
        raise ValueError("Tile list is empty: {}".format(path))
    return tile_ids


def tile_url(base_url, release, tile_id):
    tile_path = "/".join(tile_id.split("-"))
    return "{}/{}/tiles/{}/{}-obj.zip".format(
        base_url.rstrip("/"), release.strip("/"), tile_path, tile_id
    )


def read_checksums(path):
    checksums = {}
    for raw in path.expanduser().read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError("Invalid SHA-256 line: {}".format(raw))
        digest, filename = fields
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("Invalid SHA-256 digest: {}".format(digest)) from exc
        if filename in checksums:
            raise ValueError("Duplicate checksum entry: {}".format(filename))
        checksums[filename] = digest.lower()
    return checksums


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_archive(path, tile_id, full_test=True):
    expected = "{}-LoD22-3D.obj".format(tile_id)
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if not any(Path(name).name == expected for name in members):
            raise RuntimeError("Archive does not contain {}: {}".format(expected, path))
        if full_test:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError("Corrupt ZIP member {} in {}".format(bad_member, path))


def download_one(url, destination, timeout, retries):
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Point2Contour-3DBAG-reproduction/1.0"},
    )
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if partial.exists():
                partial.unlink()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with partial.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            os.replace(str(partial), str(destination))
            return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError("Download failed after {} attempts: {}".format(retries, url)) from last_error


def write_manifest(path, rows):
    fields = ["order", "tile_id", "release", "url", "file", "bytes", "sha256", "status"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    if args.retries < 1:
        raise ValueError("--retries must be positive")
    tile_ids = read_tile_ids(args.tile_list)
    checksums = {}
    if not args.skip_checksum:
        checksums = read_checksums(args.checksum_file)
        expected_names = {"{}-obj.zip".format(tile_id) for tile_id in tile_ids}
        missing = sorted(expected_names.difference(checksums))
        extra = sorted(set(checksums).difference(expected_names))
        if missing or extra:
            raise ValueError(
                "Checksum list does not match tile list; missing={}, extra={}".format(
                    missing, extra
                )
            )
    output_dir = args.output_dir.expanduser().resolve()
    if args.dry_run:
        for tile_id in tile_ids:
            print(tile_url(args.base_url, args.release, tile_id))
        print("Tiles: {}".format(len(tile_ids)))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for order, tile_id in enumerate(tile_ids, start=1):
        url = tile_url(args.base_url, args.release, tile_id)
        destination = output_dir / "{}-obj.zip".format(tile_id)
        status = "existing"
        if args.overwrite or not destination.is_file():
            print("[{}/{}] {}".format(order, len(tile_ids), tile_id))
            download_one(url, destination, args.timeout, args.retries)
            status = "downloaded"
        validate_archive(destination, tile_id, full_test=not args.skip_zip_test)
        digest = sha256_file(destination)
        expected_digest = checksums.get(destination.name)
        if expected_digest is not None and digest != expected_digest:
            raise RuntimeError(
                "SHA-256 mismatch for {}: expected {}, got {}".format(
                    destination, expected_digest, digest
                )
            )
        rows.append(
            {
                "order": order,
                "tile_id": tile_id,
                "release": args.release,
                "url": url,
                "file": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": digest,
                "status": status,
            }
        )
    manifest = output_dir / "download_manifest.csv"
    write_manifest(manifest, rows)
    print("Downloaded or verified {} tiles.".format(len(rows)))
    print("Manifest: {}".format(manifest))


if __name__ == "__main__":
    main()
