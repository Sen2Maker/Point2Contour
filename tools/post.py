#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

try:
    from .linecloud2wireframe import (
        Scheme,
        denormalize_edges,
        linecloud_to_wireframe,
        load_obj_edges,
        normalize_edges,
        save_obj_edges,
    )
except ImportError:
    from linecloud2wireframe import (
        Scheme,
        denormalize_edges,
        linecloud_to_wireframe,
        load_obj_edges,
        normalize_edges,
        save_obj_edges,
    )


TARGET_DATA_ROOT = Path("inference_results")
POINT_CLOUD_FILE = "pc.xyz"
INPUT_SEGMENT_FILE = "pre_seg.obj"
OUTPUT_WIREFRAME_FILE = "pre_wire.obj"
EDGE_THRESHOLD = 0.5

POSTPROCESS_SCHEME = Scheme(
    name="endpoint_path_strict_v2",
    cluster_eps=0.032,
    attach_eps=0.0448,
    min_samples=2,
    min_votes=2,
    min_graph_len=0.012,
    max_snap=0.04256,
    min_consistency=0.86,
    max_degree=5,
    angle_sep_deg=12.0,
    edge_support_radius=0.012,
    min_edge_support=0,
    score_support_weight=0.45,
    score_vote_weight=1.15,
    score_snap_weight=0.8,
    path_pair_max_len=0.45,
    path_support_radius=0.012,
    path_min_support=20,
    path_min_coverage=0.9,
    path_score_weight=0.45,
    path_knn=5,
)


def sample_sort_key(path):
    name = path.name
    suffix = name[6:] if name.startswith("tokyo_") else name
    return (0, int(suffix)) if suffix.isdigit() else (1, name)


def load_point_data(path):
    values = np.loadtxt(path, dtype=np.float64)
    values = np.atleast_2d(values)
    if values.size == 0 or values.shape[1] < 3:
        raise ValueError(f"Invalid point cloud: {path}")
    edge_probability = values[:, 3] if values.shape[1] >= 4 else None
    return values[:, :3], edge_probability


def normalize_points(points, center, scale):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return (points - center.reshape(1, 3)) / scale


def collect_scene_dirs(data_root, point_file, input_file):
    if (data_root / point_file).is_file() or (data_root / input_file).is_file():
        return [data_root]
    return sorted(
        (path for path in data_root.iterdir() if path.is_dir()),
        key=sample_sort_key,
    )


def process_scene(
    scene_dir,
    point_file,
    input_file,
    output_file,
    edge_point_file,
    edge_threshold,
    scheme,
):
    point_path = scene_dir / point_file
    input_path = scene_dir / input_file
    if not point_path.is_file():
        raise FileNotFoundError(f"Missing point cloud: {point_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing segment file: {input_path}")

    points, edge_probability = load_point_data(point_path)
    center = points.mean(axis=0)
    scale = float(np.linalg.norm(points - center.reshape(1, 3), axis=1).max())
    if scale <= 1e-9:
        scale = 1.0

    segments = normalize_edges(load_obj_edges(input_path), center, scale)
    edge_points = np.empty((0, 3), dtype=np.float64)
    if edge_point_file:
        edge_path = scene_dir / edge_point_file
        if edge_path.is_file() and edge_path.stat().st_size > 0:
            edge_coordinates, _ = load_point_data(edge_path)
            edge_points = normalize_points(edge_coordinates, center, scale)
    elif edge_probability is not None:
        edge_points = normalize_points(
            points[edge_probability > edge_threshold],
            center,
            scale,
        )

    wireframe, metadata = linecloud_to_wireframe(
        segments,
        edge_points,
        scheme,
    )
    wireframe_real = denormalize_edges(wireframe, center, scale)
    output_path = scene_dir / output_file
    vertex_count, edge_count = save_obj_edges(output_path, wireframe_real)
    return {
        "output_path": output_path,
        "vertices": vertex_count,
        "edges": edge_count,
        **metadata,
    }


def process_data_root(
    data_root,
    point_file=POINT_CLOUD_FILE,
    input_file=INPUT_SEGMENT_FILE,
    output_file=OUTPUT_WIREFRAME_FILE,
    edge_point_file=None,
    edge_threshold=EDGE_THRESHOLD,
    scheme=POSTPROCESS_SCHEME,
    limit=0,
    skip_existing=False,
    strict=False,
):
    data_root = Path(data_root).expanduser().resolve()
    if data_root.is_file():
        input_file = data_root.name
        data_root = data_root.parent
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    scene_dirs = collect_scene_dirs(data_root, point_file, input_file)
    if limit > 0:
        scene_dirs = scene_dirs[:limit]

    processed = 0
    skipped = 0
    failures = []
    total_vertices = 0
    total_edges = 0

    for scene_dir in scene_dirs:
        output_path = scene_dir / output_file
        if skip_existing and output_path.is_file():
            skipped += 1
            continue
        try:
            result = process_scene(
                scene_dir=scene_dir,
                point_file=point_file,
                input_file=input_file,
                output_file=output_file,
                edge_point_file=edge_point_file,
                edge_threshold=edge_threshold,
                scheme=scheme,
            )
            processed += 1
            total_vertices += result["vertices"]
            total_edges += result["edges"]
            print(
                f"{scene_dir.name}: "
                f"vertices={result['vertices']}, "
                f"edges={result['edges']}"
            )
        except Exception as error:
            failures.append((scene_dir.name, str(error)))
            print(f"Failed {scene_dir.name}: {error}")
            if strict:
                raise

    print("Post-processing completed.")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failures)}")
    print(f"Wireframe vertices: {total_vertices}")
    print(f"Wireframe edges: {total_edges}")
    return {
        "processed": processed,
        "skipped": skipped,
        "failed": len(failures),
        "vertices": total_vertices,
        "edges": total_edges,
        "failures": failures,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Point2Contour segments into compact wireframes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=TARGET_DATA_ROOT, help="Inference run, scene directory, or segment OBJ file")
    parser.add_argument("--point-file", default=POINT_CLOUD_FILE, help="Point-cloud filename in each scene")
    parser.add_argument("--input-file", default=INPUT_SEGMENT_FILE, help="Dense segment filename in each scene")
    parser.add_argument("--output-file", default=OUTPUT_WIREFRAME_FILE, help="Output wireframe filename in each scene")
    parser.add_argument("--edge-point-file", default=None, help="Optional edge-point filename for path support")
    parser.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD, help="Edge probability threshold for the fourth point-cloud column")
    parser.add_argument("--limit", type=int, default=0, help="Maximum scenes; use 0 for all scenes")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenes with an existing output file")
    parser.add_argument("--strict", action="store_true", help="Stop after the first failed scene")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be non-negative")
    if args.input_file == args.output_file:
        parser.error("input-file and output-file must be different")
    if not 0 <= args.edge_threshold <= 1:
        parser.error("edge-threshold must be in [0, 1]")
    return args


def main():
    args = parse_args()
    process_data_root(
        data_root=args.data_root,
        point_file=args.point_file,
        input_file=args.input_file,
        output_file=args.output_file,
        edge_point_file=args.edge_point_file,
        edge_threshold=args.edge_threshold,
        limit=args.limit,
        skip_existing=args.skip_existing,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
