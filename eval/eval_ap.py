"""Evaluate Building3D wireframe metrics per building and report their mean.

OBJ coordinates are evaluated in real space with a default threshold of 0.2 m.
Buildings without matched corners are excluded from the ACO mean.
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm


DISTANCE_THRESHOLD_M = 0.2
HAUSDORFF_SAMPLE_POINTS = 20
GT_FILENAME = "gt_wire.obj"

PRED_FILENAMES_TO_EVAL = {
    "pre_seg": "pre_seg.obj",
    "pre_seg_nms": "pre_seg_nms.obj",
}


METRIC_KEYS = (
    "average_corner_offset",
    "corners_precision",
    "corners_recall",
    "corners_f1",
    "edges_precision",
    "edges_recall",
    "edges_f1",
    "average_wed",
)

COUNT_KEYS = (
    "tp_corners",
    "tp_fp_corners",
    "tp_fn_corners",
    "distance",
    "tp_edges",
    "tp_fp_edges",
    "tp_fn_edges",
    "wed",
)


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def _empty_vertices():
    return np.empty((0, 3), dtype=np.float64)


def _empty_edges():
    return np.empty((0, 2), dtype=np.int64)


def _empty_edge_vertices():
    return np.empty((0, 2, 3), dtype=np.float64)


def _empty_wireframe():
    return {
        "vertices": _empty_vertices(),
        "edges": _empty_edges(),
        "edge_vertices": _empty_edge_vertices(),
    }


def _safe_div(numerator, denominator, default=0.0):
    return float(numerator) / float(denominator) if denominator else default


def _finite_mean(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def _format_value(value, decimals=4):
    return "nan" if not np.isfinite(value) else f"{float(value):.{decimals}f}"


def wireframe_from_edge_vertices(edge_vertices):
    edge_vertices = np.asarray(edge_vertices, dtype=np.float64).reshape(-1, 2, 3)
    if len(edge_vertices) == 0:
        return _empty_wireframe()

    vertices, inverse = np.unique(
        edge_vertices.reshape(-1, 3), axis=0, return_inverse=True
    )
    edges = np.sort(inverse.reshape(-1, 2), axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        return _empty_wireframe()

    edges = np.unique(edges, axis=0).astype(np.int64)
    return {
        "vertices": vertices.astype(np.float64),
        "edges": edges,
        "edge_vertices": vertices[edges].astype(np.float64),
    }


def read_wireframe_obj(filepath):
    vertices = []
    raw_edges = []

    with Path(filepath).open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "l" and len(parts) >= 3:
                indices = []
                for token in parts[1:]:
                    index = int(token.split("/")[0])
                    index = len(vertices) + index if index < 0 else index - 1
                    indices.append(index)
                raw_edges.extend(zip(indices[:-1], indices[1:]))

    if not vertices or not raw_edges:
        return _empty_wireframe()

    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    edge_vertices = [
        [vertices[first], vertices[second]]
        for first, second in raw_edges
        if 0 <= first < len(vertices)
        and 0 <= second < len(vertices)
        and first != second
    ]
    if not edge_vertices:
        return _empty_wireframe()

    return wireframe_from_edge_vertices(edge_vertices)


def hausdorff_distance_line(
    predicted_lines, target_lines, sample_points=HAUSDORFF_SAMPLE_POINTS
):
    predicted_lines = np.asarray(predicted_lines, dtype=np.float64).reshape(-1, 2, 3)
    target_lines = np.asarray(target_lines, dtype=np.float64).reshape(-1, 2, 3)
    num_predicted = len(predicted_lines)
    num_target = len(target_lines)
    if num_predicted == 0 or num_target == 0:
        return np.full((num_predicted, num_target), np.inf, dtype=np.float64)

    all_lines = np.concatenate((predicted_lines, target_lines), axis=0)
    weights = np.linspace(0, 1, sample_points).reshape(1, sample_points, 1)
    all_points = all_lines[:, 0, :][:, None, :] + weights * (
        all_lines[:, 1, :][:, None, :] - all_lines[:, 0, :][:, None, :]
    )

    distance_matrix = cdist(
        all_points[:num_predicted].reshape(-1, 3),
        all_points[num_predicted:].reshape(-1, 3),
        "euclidean",
    )
    distance_matrix = distance_matrix.reshape(
        num_predicted, sample_points, num_target, sample_points
    )
    distance_matrix = np.transpose(distance_matrix, axes=(0, 2, 1, 3))
    predicted_to_target = distance_matrix.min(-1).max(-1, keepdims=True)
    target_to_predicted = distance_matrix.min(-2).max(-1, keepdims=True)
    return np.concatenate(
        (predicted_to_target, target_to_predicted), axis=-1
    ).max(-1)


def remove_corners(corners, used_corners):
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 3)
    used_corners = np.asarray(used_corners, dtype=np.float64).reshape(-1, 3)
    if len(corners) == 0:
        return _empty_vertices()
    if len(used_corners) == 0:
        return corners.copy()

    corner_view = corners.view([("", corners.dtype)] * corners.shape[1])
    used_view = used_corners.view([("", used_corners.dtype)] * used_corners.shape[1])
    return np.setdiff1d(corner_view, used_view).view(corners.dtype).reshape(-1, 3)


def computer_edges(edge_vertices, vertices):
    indices = []
    for edge in edge_vertices:
        edge_indices = []
        for point in edge:
            matches = np.where((vertices == point).all(axis=1))[0]
            edge_indices.append(int(matches[0]) if len(matches) else -1)
        indices.append(edge_indices)
    if not indices:
        return _empty_edges()
    return np.sort(np.asarray(indices, dtype=np.int64), axis=-1)


def graph_edit_distance(pred_vertices, pred_edges, gt_vertices, gt_edges, vertex_cost):
    pred_vertices = np.asarray(pred_vertices, dtype=np.float64).reshape(-1, 3).copy()
    pred_edges = np.asarray(pred_edges, dtype=np.int64).reshape(-1, 2)
    gt_vertices = np.asarray(gt_vertices, dtype=np.float64).reshape(-1, 3)
    gt_edges = np.asarray(gt_edges, dtype=np.int64).reshape(-1, 2)

    if len(gt_vertices) == 0 or len(gt_edges) == 0:
        return 0.0 if len(pred_edges) == 0 else 1.0

    edge_cost = 0.0
    if len(pred_vertices) > 0:
        distances = cdist(pred_vertices, gt_vertices)
        vertex_cost += float(np.min(distances, axis=1).sum())
        nearest = np.argmin(distances, axis=1)
        pred_vertices[:] = gt_vertices[nearest]
        unique_pred_vertices = np.unique(pred_vertices, axis=0)
        renewed_edges = pred_edges.copy()

        for new_index, point in enumerate(unique_pred_vertices):
            old_indices = np.where((pred_vertices == point).all(axis=1))[0]
            for old_index in old_indices:
                renewed_edges[pred_edges == old_index] = new_index
        renewed_edges = np.unique(renewed_edges, axis=0)

        unmatched_gt_edges = gt_edges.copy()
        for edge in renewed_edges:
            first = np.where((gt_vertices == unique_pred_vertices[edge[0]]).all(axis=1))[0]
            second = np.where((gt_vertices == unique_pred_vertices[edge[1]]).all(axis=1))[0]
            if len(first) == 0 or len(second) == 0:
                continue
            mapped_edge = np.asarray(sorted([first[0], second[0]]))
            matched = np.where((gt_edges == mapped_edge).all(axis=1))[0]
            if len(matched):
                unmatched_gt_edges = unmatched_gt_edges[
                    np.any(unmatched_gt_edges != mapped_edge, axis=1)
                ]
            else:
                edge_cost += np.linalg.norm(
                    unique_pred_vertices[edge[0]] - unique_pred_vertices[edge[1]]
                )
    else:
        unmatched_gt_edges = gt_edges.copy()
        vertex_cost = 0.0

    for edge in unmatched_gt_edges:
        edge_cost += np.linalg.norm(gt_vertices[edge[0]] - gt_vertices[edge[1]])

    total_gt_length = sum(
        np.linalg.norm(gt_vertices[edge[0]] - gt_vertices[edge[1]])
        for edge in gt_edges
    )
    return _safe_div(edge_cost + vertex_cost, total_gt_length, default=1.0)


def _scene_metrics(
    prediction,
    target,
    distance_threshold,
    hausdorff_sample_points=HAUSDORFF_SAMPLE_POINTS,
):
    predicted_corners = prediction["vertices"]
    predicted_edges = prediction["edges"]
    predicted_edge_vertices = prediction["edge_vertices"].copy()
    target_corners = target["vertices"]
    target_edges = target["edges"]
    target_edge_vertices = target["edge_vertices"]

    if len(predicted_edges) and len(target_edges):
        edge_distance = hausdorff_distance_line(
            predicted_edge_vertices,
            target_edge_vertices,
            sample_points=hausdorff_sample_points,
        )
        predicted_indices, target_indices = linear_sum_assignment(edge_distance)
        edge_mask = (
            edge_distance[predicted_indices, target_indices] <= distance_threshold
        )
        matched_predicted_edges = predicted_edge_vertices[
            predicted_indices[edge_mask]
        ]
        matched_target_edges = target_edge_vertices[target_indices[edge_mask]]

        matched_predicted_corners = np.unique(
            matched_predicted_edges.reshape(-1, 3), axis=0
        )
        matched_target_corners = np.unique(
            matched_target_edges.reshape(-1, 3), axis=0
        )
        unmatched_predicted = remove_corners(
            predicted_corners, matched_predicted_corners
        )
        unmatched_target = remove_corners(target_corners, matched_target_corners)

        corner_distance = 0.0
        unmatched_tp = 0
        if len(unmatched_predicted) and len(unmatched_target):
            distance_matrix = cdist(unmatched_predicted, unmatched_target)
            pred_corner_indices, target_corner_indices = linear_sum_assignment(
                distance_matrix
            )
            corner_mask = (
                distance_matrix[pred_corner_indices, target_corner_indices]
                <= distance_threshold
            )
            corner_distance = float(
                distance_matrix[
                    pred_corner_indices[corner_mask],
                    target_corner_indices[corner_mask],
                ].sum()
            )
            unmatched_tp = int(corner_mask.sum())

        tp_corners = len(matched_predicted_corners) + unmatched_tp
        tp_edges = int(edge_mask.sum())

        if len(matched_predicted_corners) and len(matched_target_corners):
            corner_distance += float(
                cdist(matched_predicted_corners, matched_target_corners)
                .min(axis=1)
                .sum()
            )

        for position, pred_index in enumerate(predicted_indices[edge_mask]):
            predicted_edge_vertices[pred_index] = target_edge_vertices[
                target_indices[edge_mask][position]
            ]
        wed_vertices = np.unique(target_edge_vertices.reshape(-1, 3), axis=0)
        wed_edges = computer_edges(target_edge_vertices, wed_vertices)
        wed = graph_edit_distance(
            wed_vertices,
            wed_edges,
            target_corners.copy(),
            target_edges.copy(),
            corner_distance,
        )
    else:
        corner_distance = 0.0
        tp_corners = 0
        if len(predicted_corners) and len(target_corners):
            distance_matrix = cdist(predicted_corners, target_corners)
            predicted_indices, target_indices = linear_sum_assignment(distance_matrix)
            corner_mask = (
                distance_matrix[predicted_indices, target_indices]
                <= distance_threshold
            )
            tp_corners = int(corner_mask.sum())
            corner_distance = float(
                distance_matrix[
                    predicted_indices[corner_mask], target_indices[corner_mask]
                ].sum()
            )
        tp_edges = 0
        wed = 1.0

    pred_corner_count = len(predicted_corners)
    gt_corner_count = len(target_corners)
    pred_edge_count = len(predicted_edges)
    gt_edge_count = len(target_edges)

    corner_precision = _safe_div(tp_corners, pred_corner_count)
    corner_recall = _safe_div(tp_corners, gt_corner_count)
    corner_f1 = _safe_div(
        2.0 * corner_precision * corner_recall,
        corner_precision + corner_recall,
    )
    edge_precision = _safe_div(tp_edges, pred_edge_count)
    edge_recall = _safe_div(tp_edges, gt_edge_count)
    edge_f1 = _safe_div(
        2.0 * edge_precision * edge_recall,
        edge_precision + edge_recall,
    )

    return {
        "average_corner_offset": _safe_div(
            corner_distance, tp_corners, default=np.nan
        ),
        "corners_precision": corner_precision,
        "corners_recall": corner_recall,
        "corners_f1": corner_f1,
        "edges_precision": edge_precision,
        "edges_recall": edge_recall,
        "edges_f1": edge_f1,
        "average_wed": wed,
        "tp_corners": tp_corners,
        "tp_fp_corners": pred_corner_count,
        "tp_fn_corners": gt_corner_count,
        "distance": corner_distance,
        "tp_edges": tp_edges,
        "tp_fp_edges": pred_edge_count,
        "tp_fn_edges": gt_edge_count,
        "wed": wed,
    }


class APCalculator:
    def __init__(
        self,
        distance_thresh=DISTANCE_THRESHOLD_M,
        confidence_thresh=0.7,
        hausdorff_sample_points=HAUSDORFF_SAMPLE_POINTS,
    ):
        self.distance_thresh = distance_thresh
        self.confidence_thresh = confidence_thresh
        self.hausdorff_sample_points = hausdorff_sample_points
        self.reset()

    def compute_metrics(self, batch):
        batch_size = len(batch["predicted_vertices"])
        for index in range(batch_size):
            prediction = {
                "vertices": np.asarray(
                    batch["predicted_vertices"][index], dtype=np.float64
                ).reshape(-1, 3),
                "edges": np.asarray(
                    batch["predicted_edges"][index], dtype=np.int64
                ).reshape(-1, 2),
                "edge_vertices": np.asarray(
                    batch["pred_edges_vertices"][index], dtype=np.float64
                ).reshape(-1, 2, 3),
            }
            target = {
                "vertices": np.asarray(
                    batch["wf_vertices"][index], dtype=np.float64
                ).reshape(-1, 3),
                "edges": np.asarray(
                    batch["wf_edges"][index], dtype=np.int64
                ).reshape(-1, 2),
                "edge_vertices": np.asarray(
                    batch["wf_edges_vertices"][index], dtype=np.float64
                ).reshape(-1, 2, 3),
            }
            scene_result = _scene_metrics(
                prediction,
                target,
                self.distance_thresh,
                self.hausdorff_sample_points,
            )
            self.scene_metrics.append(scene_result)
            for key in COUNT_KEYS:
                self.ap_dict[key] += scene_result[key]
        self.batch_size = len(self.scene_metrics)

    def finalize(self):
        for key in METRIC_KEYS:
            self.ap_dict[key] = _finite_mean(
                scene[key] for scene in self.scene_metrics
            )
        return dict(self.ap_dict)

    def output_accuracy(self):
        metrics = self.finalize()
        print(f"WED: {_format_value(metrics['average_wed'])}")
        print(f"Corner offset: {_format_value(metrics['average_corner_offset'])} m")
        print(f"Corner precision: {_format_value(metrics['corners_precision'])}")
        print(f"Corner recall: {_format_value(metrics['corners_recall'])}")
        print(f"Corner F1: {_format_value(metrics['corners_f1'])}")
        print(f"Edge precision: {_format_value(metrics['edges_precision'])}")
        print(f"Edge recall: {_format_value(metrics['edges_recall'])}")
        print(f"Edge F1: {_format_value(metrics['edges_f1'])}")
        return metrics

    def reset(self):
        self.batch_size = 0
        self.scene_metrics = []
        self.ap_dict = {key: 0.0 for key in COUNT_KEYS}
        self.ap_dict.update({key: np.nan for key in METRIC_KEYS})


def make_single_scene_batch(prediction, target):
    return {
        "predicted_vertices": [prediction["vertices"]],
        "predicted_edges": [prediction["edges"]],
        "pred_edges_vertices": [prediction["edge_vertices"]],
        "wf_vertices": [target["vertices"]],
        "wf_edges": [target["edges"]],
        "wf_edges_vertices": [target["edge_vertices"]],
    }


def evaluate_obj_pair(
    prediction_path,
    target_path,
    distance_threshold,
    hausdorff_sample_points=HAUSDORFF_SAMPLE_POINTS,
):
    prediction = read_wireframe_obj(prediction_path)
    target = read_wireframe_obj(target_path)
    return _scene_metrics(
        prediction,
        target,
        distance_threshold,
        hausdorff_sample_points,
    )


def _scene_sort_key(path):
    suffix = path.name[6:] if path.name.startswith("tokyo_") else path.name
    return (0, int(suffix)) if suffix.isdigit() else (1, suffix)


def _write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_directory(
    target_dir,
    output_dir=None,
    distance_threshold=DISTANCE_THRESHOLD_M,
    hausdorff_sample_points=HAUSDORFF_SAMPLE_POINTS,
    gt_filename=GT_FILENAME,
    prediction_filenames=None,
):
    target_dir = Path(target_dir).expanduser().resolve()
    if not target_dir.is_dir():
        print(f"Target directory not found: {target_dir}")
        return 1

    output_dir = (
        target_dir
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_filenames = dict(
        PRED_FILENAMES_TO_EVAL
        if prediction_filenames is None
        else prediction_filenames
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"ap_report_{timestamp}.txt"
    scene_csv_path = output_dir / (
        f"ap_scene_metrics_{distance_threshold}m_{timestamp}.csv"
    )
    summary_csv_path = output_dir / (
        f"ap_summary_{distance_threshold}m_{timestamp}.csv"
    )

    logger = Logger(report_path)
    original_stdout = sys.stdout
    sys.stdout = logger
    try:
        print("Starting Building3D wireframe evaluation.")
        print("Aggregation: mean of per-building metrics.")
        print(f"Data directory: {target_dir}")
        print(f"Distance threshold: {distance_threshold} m", flush=True)
        print(f"Hausdorff samples per line: {hausdorff_sample_points}")

        scene_dirs = sorted(
            (path for path in target_dir.iterdir() if path.is_dir()),
            key=_scene_sort_key,
        )
        if not scene_dirs:
            print("No scene directories found.")
            return 0

        results_by_target = defaultdict(list)
        scene_rows = []
        failures = defaultdict(int)
        valid_gt_scenes = 0

        for scene_dir in tqdm(scene_dirs, desc="Evaluation", unit="building"):
            target_path = scene_dir / gt_filename
            if not target_path.is_file():
                failures["missing_gt"] += 1
                continue

            valid_gt_scenes += 1
            for target_name, prediction_filename in prediction_filenames.items():
                prediction_path = scene_dir / prediction_filename
                if not prediction_path.is_file():
                    failures[f"{target_name}:missing_prediction"] += 1
                    continue

                try:
                    metrics = evaluate_obj_pair(
                        prediction_path,
                        target_path,
                        distance_threshold,
                        hausdorff_sample_points,
                    )
                except Exception as error:
                    failures[f"{target_name}:failed"] += 1
                    print(f"Failed {scene_dir.name}/{target_name}: {error}")
                    continue

                results_by_target[target_name].append(metrics)
                row = {
                    "scene_name": scene_dir.name,
                    "target_name": target_name,
                    "prediction_filename": prediction_filename,
                    "gt_filename": gt_filename,
                    "distance_threshold_m": distance_threshold,
                }
                row.update(metrics)
                scene_rows.append(row)

        scene_fields = (
            "scene_name",
            "target_name",
            "prediction_filename",
            "gt_filename",
            "distance_threshold_m",
            *METRIC_KEYS,
            *COUNT_KEYS,
        )
        _write_csv(scene_csv_path, scene_fields, scene_rows)

        summary_rows = []
        for target_name in prediction_filenames:
            rows = results_by_target[target_name]
            summary = {
                "target_name": target_name,
                "prediction_filename": prediction_filenames[target_name],
                "gt_filename": gt_filename,
                "aggregation": "mean_of_per_building_metrics",
                "evaluated_buildings": len(rows),
                "available_gt_buildings": valid_gt_scenes,
                "missing_predictions": failures.get(
                    f"{target_name}:missing_prediction", 0
                ),
                "failed_buildings": failures.get(f"{target_name}:failed", 0),
                "distance_threshold_m": distance_threshold,
                "hausdorff_sample_points": hausdorff_sample_points,
                "valid_aco_buildings": sum(
                    np.isfinite(row["average_corner_offset"]) for row in rows
                ),
            }
            summary.update(
                {
                    key: _finite_mean(row[key] for row in rows)
                    for key in METRIC_KEYS
                }
            )
            summary_rows.append(summary)

        summary_fields = (
            "target_name",
            "prediction_filename",
            "gt_filename",
            "aggregation",
            "evaluated_buildings",
            "available_gt_buildings",
            "missing_predictions",
            "failed_buildings",
            "distance_threshold_m",
            "hausdorff_sample_points",
            "valid_aco_buildings",
            *METRIC_KEYS,
        )
        _write_csv(summary_csv_path, summary_fields, summary_rows)

        print(f"Evaluation completed: {valid_gt_scenes} buildings with GT.")
        if failures:
            print("Missing or failed inputs:")
            for name, count in sorted(failures.items()):
                if count:
                    print(f"  {name}: {count}")

        print("Summary")
        print(
            "Target | Buildings | ACO (m) | Corner P | Corner R | Corner F1 | "
            "Edge P | Edge R | Edge F1 | WED"
        )
        for row in summary_rows:
            print(
                f"{row['target_name']} | {row['evaluated_buildings']} | "
                f"{_format_value(row['average_corner_offset'])} | "
                f"{_format_value(row['corners_precision'])} | "
                f"{_format_value(row['corners_recall'])} | "
                f"{_format_value(row['corners_f1'])} | "
                f"{_format_value(row['edges_precision'])} | "
                f"{_format_value(row['edges_recall'])} | "
                f"{_format_value(row['edges_f1'])} | "
                f"{_format_value(row['average_wed'])}"
            )

        print(f"Scene metrics: {scene_csv_path.name}")
        print(f"Summary metrics: {summary_csv_path.name}")
        print(f"Report: {report_path.name}")
        return 0
    finally:
        sys.stdout = original_stdout
        logger.close()


def parse_prediction(value):
    name, separator, filename = value.partition("=")
    if not separator or not name.strip() or not filename.strip():
        raise argparse.ArgumentTypeError("prediction must use NAME=FILENAME")
    return name.strip(), filename.strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Building3D wireframe metrics per building.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-dir", required=True, help="Directory containing scene subdirectories")
    parser.add_argument("--output-dir", default=None, help="Directory for reports and CSV files")
    parser.add_argument("--distance-threshold", type=float, default=DISTANCE_THRESHOLD_M, help="Matching threshold in meters")
    parser.add_argument("--hausdorff-sample-points", type=int, default=HAUSDORFF_SAMPLE_POINTS, help="Samples per line for Hausdorff matching")
    parser.add_argument("--gt-filename", default=GT_FILENAME, help="Ground-truth filename in each scene")
    parser.add_argument("--prediction", type=parse_prediction, action="append", default=None, metavar="NAME=FILENAME", help="Prediction target; repeat for multiple files")
    args = parser.parse_args()
    if args.distance_threshold <= 0:
        parser.error("distance-threshold must be positive")
    if args.hausdorff_sample_points < 2:
        parser.error("hausdorff-sample-points must be at least 2")
    return args


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        evaluate_directory(
            target_dir=args.target_dir,
            output_dir=args.output_dir,
            distance_threshold=args.distance_threshold,
            hausdorff_sample_points=args.hausdorff_sample_points,
            gt_filename=args.gt_filename,
            prediction_filenames=args.prediction,
        )
    )
