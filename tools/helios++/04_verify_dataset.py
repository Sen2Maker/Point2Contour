#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the paired XYZ and wireframe OBJ files produced by the HELIOS++ pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--id-list",
        type=Path,
        default=None,
        help="Output list of valid paired IDs; defaults to dataset-dir/all.txt",
    )
    parser.add_argument("--expected-scenes", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Read only the first valid XYZ row per scene")
    return parser.parse_args()


def numeric_sort_key(path):
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def inspect_xyz(path, quick):
    point_count = 0
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError("{}:{} has fewer than three columns".format(path, line_number))
            coordinates = [float(value) for value in fields[:3]]
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError("{}:{} contains non-finite coordinates".format(path, line_number))
            point_count += 1
            if quick:
                break
    if point_count == 0:
        raise ValueError("Empty XYZ file: {}".format(path))
    return point_count


def resolve_obj_index(token, vertex_count):
    raw = int(token.split("/", 1)[0])
    index = raw - 1 if raw > 0 else vertex_count + raw
    if index < 0 or index >= vertex_count:
        raise IndexError("OBJ index {} is outside the vertex table".format(raw))
    return index


def inspect_wireframe(path):
    vertices = []
    line_tokens = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, raw in enumerate(stream, start=1):
            fields = raw.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError("{}:{} has an invalid vertex".format(path, line_number))
                vertex = tuple(float(value) for value in fields[1:4])
                if not all(math.isfinite(value) for value in vertex):
                    raise ValueError("{}:{} has a non-finite vertex".format(path, line_number))
                vertices.append(vertex)
            elif fields[0] == "l":
                if len(fields) < 3:
                    raise ValueError("{}:{} has an invalid line".format(path, line_number))
                line_tokens.append((line_number, fields[1:]))

    if not vertices:
        raise ValueError("Wireframe has no vertices: {}".format(path))
    edge_count = 0
    for line_number, tokens in line_tokens:
        indices = [resolve_obj_index(token, len(vertices)) for token in tokens]
        for first, second in zip(indices[:-1], indices[1:]):
            if first == second:
                raise ValueError("{}:{} contains a degenerate edge".format(path, line_number))
            edge_count += 1
    if edge_count == 0:
        raise ValueError("Wireframe has no edges: {}".format(path))
    return len(vertices), edge_count


def atomic_write_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    xyz_dir = dataset_dir / "xyz"
    wireframe_dir = dataset_dir / "wireframe"
    if not xyz_dir.is_dir() or not wireframe_dir.is_dir():
        raise FileNotFoundError(
            "Expected xyz/ and wireframe/ under {}".format(dataset_dir)
        )

    xyz_files = {path.stem: path for path in xyz_dir.glob("*.xyz")}
    wire_files = {path.stem: path for path in wireframe_dir.glob("*.obj")}
    xyz_only = sorted(set(xyz_files).difference(wire_files))
    wire_only = sorted(set(wire_files).difference(xyz_files))
    paired_ids = sorted(set(xyz_files).intersection(wire_files), key=lambda value: numeric_sort_key(xyz_files[value]))

    rows = []
    valid_ids = []
    invalid_ids = []
    for position, scene_id in enumerate(paired_ids, start=1):
        status = "ok"
        error = ""
        points = 0
        vertices = 0
        edges = 0
        try:
            points = inspect_xyz(xyz_files[scene_id], args.quick)
            vertices, edges = inspect_wireframe(wire_files[scene_id])
            valid_ids.append(scene_id)
        except Exception as exc:
            status = "invalid"
            error = "{}: {}".format(type(exc).__name__, exc)
            invalid_ids.append(scene_id)
        rows.append(
            {
                "scene_id": scene_id,
                "status": status,
                "points": points,
                "wire_vertices": vertices,
                "wire_edges": edges,
                "error": error,
            }
        )
        if position % 1000 == 0 or position == len(paired_ids):
            print("Validated {}/{} scenes.".format(position, len(paired_ids)))

    metadata_dir = dataset_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_csv = metadata_dir / "validation.csv"
    temporary = report_csv.with_suffix(report_csv.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scene_id", "status", "points", "wire_vertices", "wire_edges", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(report_csv))

    id_list_path = args.id_list.expanduser().resolve() if args.id_list else dataset_dir / "all.txt"
    id_list_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(id_list_path, "".join("{}\n".format(scene_id) for scene_id in valid_ids))

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "quick": args.quick,
        "xyz_files": len(xyz_files),
        "wireframe_files": len(wire_files),
        "paired_files": len(paired_ids),
        "valid_scenes": len(valid_ids),
        "invalid_scenes": invalid_ids,
        "xyz_without_wireframe": xyz_only,
        "wireframe_without_xyz": wire_only,
        "expected_scenes": args.expected_scenes,
    }
    report_path = metadata_dir / "validation.json"
    atomic_write_text(report_path, json.dumps(report, indent=2, ensure_ascii=True) + "\n")

    failed = bool(invalid_ids or xyz_only or wire_only)
    if args.expected_scenes is not None and len(valid_ids) != args.expected_scenes:
        failed = True
    print("Valid paired scenes: {}".format(len(valid_ids)))
    print("ID list: {}".format(id_list_path))
    print("Report: {}".format(report_path))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
