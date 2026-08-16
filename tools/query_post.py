#!/usr/bin/env python3

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from .linecloud2wireframe import load_obj_edges, save_obj_edges
except ImportError:
    from linecloud2wireframe import load_obj_edges, save_obj_edges


TARGET_DATA_ROOT = Path("res_pre")
QUERY_FILE = "raw_topk.obj"
INPUT_FILES = ("raw_topk.obj", "pre_seg.obj", "pre_seg_nms.obj")
OUTPUT_SUFFIX = "_filtered"
RAW_CORNER_OUTPUT_FILE = "corners_raw.ply"
CORNER_OUTPUT_FILE = "corners_filtered.ply"
QUERY_INDEX_FILE = "query_indices.txt"
SCENE_REPORT_FILE = "query_post.json"
SUMMARY_FILE = "query_post_summary.csv"


@dataclass(frozen=True)
class QueryScheme:
    name: str = "reciprocal_incoming_v1"
    rays_per_query: int = 6
    reciprocal_tolerance_m: float = 0.30
    reciprocal_min_neighbors: int = 1
    incoming_tolerance_m: float = 0.20
    incoming_min_sources: int = 3
    incoming_min_directions: int = 2
    direction_separation_deg: float = 20.0


POSTPROCESS_SCHEME = QueryScheme()


def sample_sort_key(path):
    name = path.name
    suffix = name[6:] if name.startswith("tokyo_") else name
    return (0, int(suffix)) if suffix.isdigit() else (1, name)


def parse_name_list(values):
    output = []
    seen = set()
    for value in values:
        for name in str(value).replace(",", " ").replace(";", " ").split():
            if name and name not in seen:
                output.append(name)
                seen.add(name)
    return output


def parse_id_list(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    text = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else value
    return parse_name_list([text])


def output_name(input_name, suffix):
    path = Path(input_name)
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def reconstruct_queries(lines, rays_per_query):
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 2, 3)
    if len(lines) == 0:
        raise ValueError("The query file contains no lines")
    corners = []
    ray_source = np.empty(len(lines), dtype=np.int64)
    ray_position = np.empty(len(lines), dtype=np.int64)
    current_corner = None
    current_count = 0
    current_source = -1
    for line_index, start in enumerate(lines[:, 0]):
        same_source = (
            current_corner is not None
            and np.array_equal(start, current_corner)
            and current_count < rays_per_query
        )
        if not same_source:
            corners.append(start.copy())
            current_corner = start.copy()
            current_count = 0
            current_source += 1
        ray_source[line_index] = current_source
        ray_position[line_index] = current_count
        current_count += 1
    query_sizes = np.bincount(ray_source, minlength=len(corners))
    invalid = np.flatnonzero(query_sizes != rays_per_query)
    if len(invalid):
        preview = ", ".join(
            f"{int(index)}:{int(query_sizes[index])}" for index in invalid[:5]
        )
        raise ValueError(
            f"Every raw query must contain exactly {rays_per_query} consecutive "
            f"rays; invalid query sizes: {preview}"
        )
    return (
        np.asarray(corners, dtype=np.float64),
        ray_source,
        ray_position,
        query_sizes,
    )


def count_distinct_directions(directions, scores, minimum_angle_deg):
    kept = []
    for index in np.argsort(-scores, kind="stable"):
        direction = directions[index]
        if all(
            np.degrees(np.arccos(np.clip(np.dot(direction, old), -1.0, 1.0)))
            >= minimum_angle_deg
            for old in kept
        ):
            kept.append(direction)
    return len(kept)


def select_queries(lines, scheme=POSTPROCESS_SCHEME):
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 2, 3)
    corners, ray_source, ray_position, query_sizes = reconstruct_queries(
        lines,
        scheme.rays_per_query,
    )
    query_count = len(corners)
    if query_count < 2:
        raise ValueError("At least two corner queries are required")

    endpoints = lines[:, 1]
    ray_count = len(endpoints)
    tree = cKDTree(corners)
    distances, owners = tree.query(endpoints, k=min(3, query_count))
    distances = np.asarray(distances).reshape(ray_count, -1)
    owners = np.asarray(owners).reshape(ray_count, -1)
    nonself = owners != ray_source[:, None]
    has_nonself = np.any(nonself, axis=1)
    first_nonself = np.argmax(nonself, axis=1)
    target = owners[np.arange(ray_count), first_nonself]
    forward_error = distances[np.arange(ray_count), first_nonself]
    target = np.where(has_nonself, target, -1)
    forward_error = np.where(has_nonself, forward_error, np.inf)

    padded_endpoints = np.full(
        (query_count, scheme.rays_per_query, 3),
        np.inf,
        dtype=np.float64,
    )
    valid_positions = ray_position < scheme.rays_per_query
    padded_endpoints[ray_source[valid_positions], ray_position[valid_positions]] = endpoints[
        valid_positions
    ]
    reverse_error = np.full(ray_count, np.inf, dtype=np.float64)
    valid_target = target >= 0
    reverse_error[valid_target] = np.linalg.norm(
        padded_endpoints[target[valid_target]]
        - corners[ray_source[valid_target], None, :],
        axis=2,
    ).min(axis=1)

    vectors = endpoints - corners[ray_source]
    directions = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
    reciprocal_ray = (
        (target >= 0)
        & (forward_error <= scheme.reciprocal_tolerance_m)
        & (reverse_error <= scheme.reciprocal_tolerance_m)
    )
    reciprocal_mask = np.zeros(query_count, dtype=bool)
    for source in range(query_count):
        rays = np.flatnonzero(reciprocal_ray & (ray_source == source))
        if len(rays) == 0:
            continue
        best_by_target = {}
        for ray in rays:
            candidate = int(target[ray])
            error = float(forward_error[ray] + reverse_error[ray])
            old = best_by_target.get(candidate)
            if old is None or error < old[1]:
                best_by_target[candidate] = (int(ray), error)
        selected_rays = np.asarray(
            [item[0] for item in best_by_target.values()],
            dtype=np.int64,
        )
        quality = np.exp(
            -np.asarray([item[1] for item in best_by_target.values()])
            / (2.0 * scheme.reciprocal_tolerance_m)
        )
        direction_count = count_distinct_directions(
            directions[selected_rays],
            quality,
            scheme.direction_separation_deg,
        )
        if (
            len(best_by_target) >= scheme.reciprocal_min_neighbors
            and direction_count >= scheme.reciprocal_min_neighbors
        ):
            reciprocal_mask[source] = True

    incoming_mask = np.zeros(query_count, dtype=bool)
    vote_rays = np.flatnonzero(
        (target >= 0) & (forward_error <= scheme.incoming_tolerance_m)
    )
    for candidate in np.unique(target[vote_rays]):
        candidate_rays = vote_rays[target[vote_rays] == candidate]
        best_by_source = {}
        for ray in candidate_rays:
            source = int(ray_source[ray])
            error = float(forward_error[ray])
            old = best_by_source.get(source)
            if old is None or error < old[1]:
                best_by_source[source] = (int(ray), error)
        if len(best_by_source) < scheme.incoming_min_sources:
            continue
        selected_rays = np.asarray(
            [item[0] for item in best_by_source.values()],
            dtype=np.int64,
        )
        quality = np.exp(
            -np.asarray([item[1] for item in best_by_source.values()])
            / scheme.incoming_tolerance_m
        )
        direction_count = count_distinct_directions(
            -directions[selected_rays],
            quality,
            scheme.direction_separation_deg,
        )
        if direction_count >= scheme.incoming_min_directions:
            incoming_mask[int(candidate)] = True

    selected_mask = reciprocal_mask | incoming_mask
    return {
        "corners": corners,
        "ray_source": ray_source,
        "query_sizes": query_sizes,
        "selected_mask": selected_mask,
        "selected_indices": np.flatnonzero(selected_mask),
        "reciprocal_mask": reciprocal_mask,
        "incoming_mask": incoming_mask,
    }


def save_corner_ply(path, corners):
    path.parent.mkdir(parents=True, exist_ok=True)
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(corners)}\n")
        stream.write("property double x\nproperty double y\nproperty double z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for point in corners:
            stream.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} 255 0 127\n"
            )


def write_indices(path, indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{int(index)}\n" for index in indices)
    path.write_text(text, encoding="utf-8")


def assignment_summary(distances):
    distances = np.asarray(distances, dtype=np.float64)
    if len(distances) == 0:
        return {
            "assignment_median_m": math.nan,
            "assignment_p95_m": math.nan,
            "assignment_max_m": math.nan,
        }
    return {
        "assignment_median_m": float(np.median(distances)),
        "assignment_p95_m": float(np.percentile(distances, 95)),
        "assignment_max_m": float(np.max(distances)),
    }


def process_target(
    scene_dir,
    output_dir,
    input_name,
    query_file,
    output_suffix,
    selection,
    query_tree,
    skip_existing,
):
    input_path = scene_dir / input_name
    output_relative = output_name(input_name, output_suffix)
    output_path = output_dir / output_relative
    base = {
        "input_file": input_name,
        "output_file": output_relative,
    }
    if not input_path.is_file():
        return {**base, "status": "missing_input"}
    if skip_existing and output_path.is_file():
        return {**base, "status": "skipped_existing"}

    lines = load_obj_edges(input_path)
    if input_name == query_file:
        if len(lines) != len(selection["ray_source"]):
            raise ValueError(
                f"Query line count changed while reading {input_path}: "
                f"{len(lines)} != {len(selection['ray_source'])}"
            )
        source_queries = selection["ray_source"]
        source_distances = np.zeros(len(lines), dtype=np.float64)
    elif len(lines):
        source_distances, source_queries = query_tree.query(lines[:, 0])
    else:
        source_queries = np.empty(0, dtype=np.int64)
        source_distances = np.empty(0, dtype=np.float64)
    retained = selection["selected_mask"][source_queries]
    filtered = lines[retained]
    output_vertices, output_edges = save_obj_edges(output_path, filtered)
    return {
        **base,
        "status": "ok",
        "input_lines": int(len(lines)),
        "retained_lines": int(len(filtered)),
        "retained_fraction": float(np.mean(retained)) if len(retained) else 0.0,
        "output_vertices": output_vertices,
        "output_edges": output_edges,
        **assignment_summary(source_distances),
    }


def process_scene(
    scene_dir,
    output_dir,
    query_file=QUERY_FILE,
    input_files=INPUT_FILES,
    output_suffix=OUTPUT_SUFFIX,
    scheme=POSTPROCESS_SCHEME,
    skip_existing=False,
    strict=False,
):
    query_path = scene_dir / query_file
    if not query_path.is_file():
        raise FileNotFoundError(f"Missing query file: {query_path}")
    query_lines = load_obj_edges(query_path)
    selection = select_queries(query_lines, scheme)
    query_tree = cKDTree(selection["corners"])
    selected_corners = selection["corners"][selection["selected_mask"]]

    save_corner_ply(output_dir / RAW_CORNER_OUTPUT_FILE, selection["corners"])
    save_corner_ply(output_dir / CORNER_OUTPUT_FILE, selected_corners)
    write_indices(output_dir / QUERY_INDEX_FILE, selection["selected_indices"])

    target_rows = []
    for input_name in input_files:
        try:
            row = process_target(
                scene_dir=scene_dir,
                output_dir=output_dir,
                input_name=input_name,
                query_file=query_file,
                output_suffix=output_suffix,
                selection=selection,
                query_tree=query_tree,
                skip_existing=skip_existing,
            )
        except Exception as error:
            row = {
                "input_file": input_name,
                "output_file": output_name(input_name, output_suffix),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
            if strict:
                raise
        if strict and row["status"] == "missing_input":
            raise FileNotFoundError(f"Missing input file: {scene_dir / input_name}")
        target_rows.append(row)

    report = {
        "scene_id": scene_dir.name,
        "method": scheme.name,
        "selection_uses_ground_truth": False,
        "query_file": query_file,
        "parameters": asdict(scheme),
        "selection": {
            "raw_queries": int(len(selection["corners"])),
            "reciprocal_queries": int(selection["reciprocal_mask"].sum()),
            "incoming_queries": int(selection["incoming_mask"].sum()),
            "intersection_queries": int(
                np.sum(selection["reciprocal_mask"] & selection["incoming_mask"])
            ),
            "retained_queries": int(selection["selected_mask"].sum()),
            "minimum_rays_per_query": int(selection["query_sizes"].min()),
            "maximum_rays_per_query": int(selection["query_sizes"].max()),
        },
        "outputs": target_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SCENE_REPORT_FILE).write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def collect_scene_dirs(data_root, query_file, ids=None):
    if (data_root / query_file).is_file():
        scenes = [data_root]
    else:
        scenes = sorted(
            (path for path in data_root.iterdir() if path.is_dir()),
            key=sample_sort_key,
        )
    if ids is None:
        return scenes
    wanted = set(ids)
    return [scene for scene in scenes if scene.name in wanted]


def flatten_report(report):
    selection = report["selection"]
    rows = []
    for output in report["outputs"]:
        rows.append(
            {
                "scene_id": report["scene_id"],
                "method": report["method"],
                "query_file": report["query_file"],
                **selection,
                **output,
            }
        )
    return rows


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not fieldnames:
            stream.write("")
            return
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_data_root(
    data_root,
    query_file=QUERY_FILE,
    input_files=INPUT_FILES,
    output_suffix=OUTPUT_SUFFIX,
    output_root=None,
    ids=None,
    limit=0,
    skip_existing=False,
    strict=False,
    scheme=POSTPROCESS_SCHEME,
):
    data_root = Path(data_root).expanduser().resolve()
    if data_root.is_file():
        input_files = (data_root.name,)
        data_root = data_root.parent
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    output_root = (
        data_root
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    scene_dirs = collect_scene_dirs(data_root, query_file, ids)
    if limit > 0:
        scene_dirs = scene_dirs[:limit]

    rows = []
    processed = 0
    failed = 0
    single_scene_root = len(scene_dirs) == 1 and scene_dirs[0] == data_root
    for scene_dir in scene_dirs:
        output_dir = (
            output_root
            if output_root == data_root and single_scene_root
            else output_root / scene_dir.name
        )
        try:
            report = process_scene(
                scene_dir=scene_dir,
                output_dir=output_dir,
                query_file=query_file,
                input_files=input_files,
                output_suffix=output_suffix,
                scheme=scheme,
                skip_existing=skip_existing,
                strict=strict,
            )
            rows.extend(flatten_report(report))
            processed += 1
            result_text = ", ".join(
                f"{row['input_file']}={row.get('input_lines', 0)}->"
                f"{row.get('retained_lines', 0)}"
                for row in report["outputs"]
                if row["status"] == "ok"
            )
            print(
                f"{scene_dir.name}: queries="
                f"{report['selection']['raw_queries']}->"
                f"{report['selection']['retained_queries']}; {result_text}"
            )
        except Exception as error:
            failed += 1
            rows.append(
                {
                    "scene_id": scene_dir.name,
                    "method": scheme.name,
                    "query_file": query_file,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"Failed {scene_dir.name}: {error}")
            if strict:
                raise

    summary_path = output_root / SUMMARY_FILE
    write_summary(summary_path, rows)
    print("Query post-processing completed.")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Summary: {summary_path}")
    return {
        "processed": processed,
        "failed": failed,
        "summary_path": summary_path,
        "rows": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter Point2Contour query corners and their derived line clouds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=TARGET_DATA_ROOT,
        help="Inference run, scene directory, or line-cloud OBJ file",
    )
    parser.add_argument(
        "--query-file",
        default=QUERY_FILE,
        help="Raw Top-K query line file in each scene",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=list(INPUT_FILES),
        help="Files to filter; spaces or commas are accepted",
    )
    parser.add_argument(
        "--output-suffix",
        default=OUTPUT_SUFFIX,
        help="Suffix inserted before each input extension",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional separate output root; defaults to in-place output",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Scene IDs separated by commas/spaces, or a text file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum scenes; use 0 for all scenes",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip target files whose outputs already exist",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop after the first missing or invalid scene",
    )
    args = parser.parse_args()
    args.input_files = parse_name_list(args.input_files)
    args.ids = parse_id_list(args.ids)
    if args.limit < 0:
        parser.error("limit must be non-negative")
    if not args.input_files:
        parser.error("input-files must contain at least one filename")
    if not args.output_suffix:
        parser.error("output-suffix must not be empty")
    if any(output_name(name, args.output_suffix) == name for name in args.input_files):
        parser.error("output filenames must differ from input filenames")
    return args


def main():
    args = parse_args()
    process_data_root(
        data_root=args.data_root,
        query_file=args.query_file,
        input_files=args.input_files,
        output_suffix=args.output_suffix,
        output_root=args.output_root,
        ids=args.ids,
        limit=args.limit,
        skip_existing=args.skip_existing,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
