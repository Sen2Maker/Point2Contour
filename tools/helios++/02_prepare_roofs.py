#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TILE_LIST = SCRIPT_DIR / "woerden_tiles_v20250903.txt"
DEFAULT_RELEASE = "v20250903"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract one roof mesh and one GT wireframe per building from 3DBAG OBJ tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zip-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-list", type=Path, default=DEFAULT_TILE_LIST)
    parser.add_argument("--source-release", default=DEFAULT_RELEASE)
    parser.add_argument("--lod", default="LoD22")
    parser.add_argument("--roof-material", default="1")
    parser.add_argument("--crease-angle", type=float, default=4.0)
    parser.add_argument("--min-faces", type=int, default=1)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument(
        "--max-bbox-diagonal",
        type=float,
        default=1000.0,
        help="Reject malformed roof meshes above this size in meters; use 0 to disable",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tile_ids(path):
    tile_ids = []
    seen = set()
    for raw in path.expanduser().read_text(encoding="utf-8").splitlines():
        tile_id = raw.split("#", 1)[0].strip()
        if not tile_id:
            continue
        if tile_id in seen:
            raise ValueError("Duplicate tile ID: {}".format(tile_id))
        seen.add(tile_id)
        tile_ids.append(tile_id)
    if not tile_ids:
        raise ValueError("Tile list is empty: {}".format(path))
    return tile_ids


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_vertex_index(token, vertex_count):
    raw = int(token.split("/", 1)[0])
    index = raw - 1 if raw > 0 else vertex_count + raw
    if index < 0 or index >= vertex_count:
        raise IndexError("OBJ vertex index {} is outside [1, {}]".format(raw, vertex_count))
    return index


def iter_roof_objects(archive_path, member_name, roof_material):
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as binary_stream:
            stream = io.TextIOWrapper(binary_stream, encoding="utf-8", errors="strict")
            vertices = []
            object_name = "NO_OBJECT"
            material = None
            faces = []
            faces_started = False

            for raw_line in stream:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                record = fields[0]

                if record == "v" and len(fields) >= 4:
                    if faces_started:
                        raise RuntimeError(
                            "A vertex appears after face records in {}; this layout is unsupported".format(
                                archive_path
                            )
                        )
                    vertices.append(tuple(float(value) for value in fields[1:4]))
                elif record == "o":
                    if faces:
                        yield object_name, faces, vertices
                    object_name = line[2:].strip() or "NO_OBJECT"
                    faces = []
                elif record == "usemtl" and len(fields) >= 2:
                    material = fields[1]
                elif record == "f" and len(fields) >= 4:
                    faces_started = True
                    if material == roof_material:
                        face = tuple(resolve_vertex_index(token, len(vertices)) for token in fields[1:])
                        if len(set(face)) >= 3:
                            faces.append(face)

            if faces:
                yield object_name, faces, vertices


def find_obj_member(archive_path, tile_id, lod):
    expected = "{}-{}-3D.obj".format(tile_id, lod)
    with zipfile.ZipFile(archive_path) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == expected]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one {} in {}, found {}".format(expected, archive_path, len(matches))
        )
    return matches[0]


def vector_subtract(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def vector_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vector_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def face_normal(face, vertices):
    p0, p1, p2 = (vertices[face[index]] for index in range(3))
    normal = vector_cross(vector_subtract(p1, p0), vector_subtract(p2, p0))
    length = math.sqrt(vector_dot(normal, normal))
    if length < 1e-12:
        return None
    return tuple(value / length for value in normal)


def extract_wire_edges(vertices, faces, angle_degrees):
    edge_to_faces = defaultdict(list)
    normals = []
    for face_index, face in enumerate(faces):
        normals.append(face_normal(face, vertices))
        for index, first in enumerate(face):
            second = face[(index + 1) % len(face)]
            if first != second:
                edge_to_faces[tuple(sorted((first, second)))].append(face_index)

    threshold = math.cos(math.radians(angle_degrees))
    output = []
    for edge, adjacent in edge_to_faces.items():
        if len(adjacent) != 2:
            output.append(edge)
            continue
        first_normal = normals[adjacent[0]]
        second_normal = normals[adjacent[1]]
        if first_normal is None or second_normal is None:
            output.append(edge)
            continue
        cosine = min(1.0, max(-1.0, abs(vector_dot(first_normal, second_normal))))
        if cosine < threshold:
            output.append(edge)
    return sorted(set(output))


def used_indices_from_faces(faces):
    return sorted({index for face in faces for index in face})


def bbox_from_indices(vertices, indices):
    axes = [[vertices[index][axis] for index in indices] for axis in range(3)]
    lower = tuple(min(values) for values in axes)
    upper = tuple(max(values) for values in axes)
    diagonal = math.sqrt(sum((upper[i] - lower[i]) ** 2 for i in range(3)))
    return lower, upper, diagonal


def write_mesh(path, scene_id, source_name, vertices, faces):
    used = used_indices_from_faces(faces)
    remap = {old: new for new, old in enumerate(used, start=1)}
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Derived from 3DBAG object {}\n".format(source_name))
        stream.write("o {}\n".format(scene_id))
        for old in used:
            stream.write("v {:.10f} {:.10f} {:.10f}\n".format(*vertices[old]))
        for face in faces:
            stream.write("f {}\n".format(" ".join(str(remap[index]) for index in face)))
    return len(used)


def write_wireframe(path, scene_id, source_name, vertices, edges):
    used = sorted({index for edge in edges for index in edge})
    remap = {old: new for new, old in enumerate(used, start=1)}
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Derived from 3DBAG object {}\n".format(source_name))
        stream.write("o {}\n".format(scene_id))
        for old in used:
            stream.write("v {:.10f} {:.10f} {:.10f}\n".format(*vertices[old]))
        for first, second in edges:
            stream.write("l {} {}\n".format(remap[first], remap[second]))
    return len(used)


def atomic_csv(path, rows, fieldnames):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    if args.min_faces < 1 or args.start_id < 0:
        raise ValueError("--min-faces must be positive and --start-id must be non-negative")
    if not 0 <= args.crease_angle <= 180:
        raise ValueError("--crease-angle must be in [0, 180]")

    zip_dir = args.zip_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metadata_dir = output_dir / "metadata"
    mesh_dir = output_dir / "roof_mesh"
    wireframe_dir = output_dir / "wireframe"
    manifest_path = metadata_dir / "buildings.csv"
    generated_outputs_exist = manifest_path.exists() or any(mesh_dir.glob("*.obj")) or any(
        wireframe_dir.glob("*.obj")
    )
    if generated_outputs_exist and not args.overwrite:
        raise FileExistsError(
            "Generated output already exists; use --overwrite or a new directory: {}".format(
                output_dir
            )
        )
    existing_xyz = output_dir / "xyz"
    if args.overwrite and any(existing_xyz.glob("*.xyz")):
        raise RuntimeError(
            "Refusing to replace roof geometry beside existing XYZ files. Use a new output directory to avoid mixing simulation and GT versions."
        )
    if args.overwrite:
        for directory in (mesh_dir, wireframe_dir):
            if directory.is_dir():
                shutil.rmtree(str(directory))
    for directory in (metadata_dir, mesh_dir, wireframe_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tile_ids = read_tile_ids(args.tile_list)
    source_archives = []
    for tile_id in tile_ids:
        archive_path = zip_dir / "{}-obj.zip".format(tile_id)
        if not archive_path.is_file():
            raise FileNotFoundError("Missing 3DBAG archive: {}".format(archive_path))
        source_archives.append(
            {
                "tile_id": tile_id,
                "file": archive_path.name,
                "bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
            }
        )
    rows = []
    scene_id = args.start_id
    valid_ids = []

    for tile_order, tile_id in enumerate(tile_ids, start=1):
        archive_path = zip_dir / "{}-obj.zip".format(tile_id)
        member_name = find_obj_member(archive_path, tile_id, args.lod)
        tile_count = 0
        for source_name, faces, vertices in iter_roof_objects(
            archive_path, member_name, args.roof_material
        ):
            if len(faces) < args.min_faces:
                continue
            current_id = scene_id
            scene_id += 1
            tile_count += 1
            used = used_indices_from_faces(faces)
            lower, upper, diagonal = bbox_from_indices(vertices, used)
            edges = extract_wire_edges(vertices, faces, args.crease_angle)

            status = "ok"
            if args.max_bbox_diagonal > 0 and diagonal > args.max_bbox_diagonal:
                status = "invalid_bbox"
            elif not edges:
                status = "no_wireframe_edges"

            mesh_relative = ""
            wire_relative = ""
            wire_vertices = 0
            if status == "ok":
                mesh_path = mesh_dir / "{}.obj".format(current_id)
                wire_path = wireframe_dir / "{}.obj".format(current_id)
                mesh_vertices = write_mesh(
                    mesh_path, current_id, source_name, vertices, faces
                )
                wire_vertices = write_wireframe(
                    wire_path, current_id, source_name, vertices, edges
                )
                mesh_relative = mesh_path.relative_to(output_dir).as_posix()
                wire_relative = wire_path.relative_to(output_dir).as_posix()
                valid_ids.append(str(current_id))
            else:
                mesh_vertices = len(used)

            rows.append(
                {
                    "scene_id": current_id,
                    "tile_order": tile_order,
                    "tile_id": tile_id,
                    "source_object": source_name,
                    "lod": args.lod,
                    "roof_material": args.roof_material,
                    "roof_faces": len(faces),
                    "mesh_vertices": mesh_vertices,
                    "wire_vertices": wire_vertices,
                    "wire_edges": len(edges),
                    "xmin": "{:.6f}".format(lower[0]),
                    "xmax": "{:.6f}".format(upper[0]),
                    "ymin": "{:.6f}".format(lower[1]),
                    "ymax": "{:.6f}".format(upper[1]),
                    "zmin": "{:.6f}".format(lower[2]),
                    "zmax": "{:.6f}".format(upper[2]),
                    "bbox_diagonal": "{:.6f}".format(diagonal),
                    "status": status,
                    "roof_mesh": mesh_relative,
                    "wireframe": wire_relative,
                }
            )
        print("[{}/{}] {}: {} buildings".format(tile_order, len(tile_ids), tile_id, tile_count))

    fields = [
        "scene_id",
        "tile_order",
        "tile_id",
        "source_object",
        "lod",
        "roof_material",
        "roof_faces",
        "mesh_vertices",
        "wire_vertices",
        "wire_edges",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
        "bbox_diagonal",
        "status",
        "roof_mesh",
        "wireframe",
    ]
    atomic_csv(manifest_path, rows, fields)
    download_manifest = zip_dir / "download_manifest.csv"
    if download_manifest.is_file():
        shutil.copy2(str(download_manifest), str(metadata_dir / "source_download_manifest.csv"))
    (metadata_dir / "prepared_ids.txt").write_text(
        "".join("{}\n".format(scene) for scene in valid_ids), encoding="utf-8"
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_release": args.source_release,
        "tile_list": str(args.tile_list.expanduser().resolve()),
        "tile_list_sha256": sha256_file(args.tile_list.expanduser().resolve()),
        "tile_ids": tile_ids,
        "source_archives": source_archives,
        "lod": args.lod,
        "roof_material": args.roof_material,
        "crease_angle_degrees": args.crease_angle,
        "min_faces": args.min_faces,
        "start_id": args.start_id,
        "max_bbox_diagonal_m": args.max_bbox_diagonal,
        "scene_count_total": len(rows),
        "scene_count_valid": len(valid_ids),
    }
    (metadata_dir / "preparation.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    rejected = len(rows) - len(valid_ids)
    print("Prepared {} buildings; {} accepted and {} rejected.".format(len(rows), len(valid_ids), rejected))
    print("Dataset directory: {}".format(output_dir))


if __name__ == "__main__":
    main()
