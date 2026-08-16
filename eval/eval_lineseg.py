import argparse
import os
import sys
import csv
import unicodedata
import numpy as np
from datetime import datetime
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


class Logger(object):

    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def pad_str(s, total_width):

    w = sum(2 if unicodedata.east_asian_width(c) in "FW" else 1 for c in str(s))
    return str(s) + " " * max(0, total_width - w)


def is_nan_scalar(v):

    return isinstance(v, (float, np.floating)) and np.isnan(v)


def fmt_float(v, decimals=2):

    if is_nan_scalar(v):
        return "nan"
    return f"{float(v):.{decimals}f}"


def read_lines_from_obj(filepath):

    if not os.path.exists(filepath):
        return np.empty((0, 2, 3), dtype=np.float64)

    vertices = []
    lines = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])

            elif parts[0] == "l":
                if len(parts) < 3:
                    continue

                idx1 = int(parts[1]) - 1
                idx2 = int(parts[2]) - 1

                if 0 <= idx1 < len(vertices) and 0 <= idx2 < len(vertices):
                    lines.append([vertices[idx1], vertices[idx2]])

    if len(lines) == 0:
        return np.empty((0, 2, 3), dtype=np.float64)

    return np.asarray(lines, dtype=np.float64)


def parse_corners_from_lines(lines):

    if len(lines) == 0:
        return np.empty((0, 3), dtype=np.float64)

    corners = np.unique(np.round(np.asarray(lines).reshape(-1, 3), 4), axis=0)
    return corners.astype(np.float64)


class LineCloudEvaluator:

    def __init__(self, name="Default_Stage", thresholds=None, density=None):
        if thresholds is None:
            thresholds = [0.1, 0.25, 0.5]

        self.name = name
        self.thresholds = thresholds

        self.density = density if density is not None else min(self.thresholds) / 2.0

        self.reset()

    def reset(self):

        self.num_samples = 0
        self.acc_metrics = {thr: defaultdict(float) for thr in self.thresholds}
        self.valid_counts = {thr: defaultdict(int) for thr in self.thresholds}

    def _sample_points_adaptive(self, lines):

        if len(lines) == 0:
            return np.empty((0, 3), dtype=np.float64)

        lines = np.asarray(lines, dtype=np.float64)
        pts = []

        for start, end in lines:
            length = np.linalg.norm(end - start)

            if length < 1e-6:
                pts.append(start.reshape(1, -1))
                continue

            num = max(3, int(np.ceil(length / self.density)) + 1)

            t = np.linspace(0, 1, num).reshape(-1, 1)
            pts.append((1 - t) * start + t * end)

        return np.vstack(pts)

    def _point_to_segment_distance(self, query_pts, target_lines, chunk_size=2000):

        if len(query_pts) == 0:
            return np.empty((0,), dtype=np.float64)

        if len(target_lines) == 0:
            return np.full(len(query_pts), np.inf, dtype=np.float64)

        query_pts = np.asarray(query_pts, dtype=np.float64)
        target_lines = np.asarray(target_lines, dtype=np.float64)

        A = target_lines[:, 0]
        B = target_lines[:, 1]
        AB = B - A

        denom = np.sum(AB ** 2, axis=1) + 1e-8
        min_dists = np.zeros(len(query_pts), dtype=np.float64)

        for i in range(0, len(query_pts), chunk_size):
            chunk_pts = query_pts[i: i + chunk_size]

            AP = chunk_pts[:, None] - A[None, :]
            proj = np.sum(AP * AB[None, :], axis=2)
            t = np.clip(proj / denom[None, :], 0.0, 1.0)

            proj_pt = A[None, :] + t[:, :, None] * AB[None, :]
            dists = np.linalg.norm(chunk_pts[:, None] - proj_pt, axis=2)

            min_dists[i: i + chunk_size] = dists.min(axis=1)

        return min_dists

    def _compute_corner_metrics(self, pred_corners, gt_corners, thresh):

        pred_empty = len(pred_corners) == 0
        gt_empty = len(gt_corners) == 0

        if pred_empty and gt_empty:
            return {"CR": 1.0, "CD": np.nan}

        if pred_empty and not gt_empty:
            return {"CR": 0.0, "CD": np.nan}

        if (not pred_empty) and gt_empty:
            return {"CR": 0.0, "CD": np.nan}

        dist_mat = cdist(pred_corners, gt_corners)
        row, col = linear_sum_assignment(dist_mat)

        dists = dist_mat[row, col]
        mask = dists <= thresh
        tp = int(np.sum(mask))

        recall = tp / len(gt_corners)
        distance = (
            float(np.sum(dists) / len(gt_corners))
            if len(dists) > 0
            else np.nan
        )
        return {"CR": recall, "CD": distance}

    def evaluate_scene(self, pred_lines, gt_lines):

        pred_lines = np.asarray(pred_lines, dtype=np.float64)
        gt_lines = np.asarray(gt_lines, dtype=np.float64)

        pred_empty = len(pred_lines) == 0
        gt_empty = len(gt_lines) == 0

        pred_corners = parse_corners_from_lines(pred_lines)
        gt_corners = parse_corners_from_lines(gt_lines)

        pts_pred = self._sample_points_adaptive(pred_lines) if not pred_empty else np.empty((0, 3), dtype=np.float64)
        pts_gt = self._sample_points_adaptive(gt_lines) if not gt_empty else np.empty((0, 3), dtype=np.float64)

        dist_P2L = self._point_to_segment_distance(pts_pred, gt_lines)
        dist_G2P = self._point_to_segment_distance(pts_gt, pred_lines)

        num_lines = len(pred_lines)

        scene_res = {}

        for thresh in self.thresholds:
            corner_metrics = self._compute_corner_metrics(
                pred_corners, gt_corners, thresh
            )

            if pred_empty and gt_empty:
                cov_p = 1.0
                cov_r = 1.0
                line_f1 = 1.0
                mean_p2l = np.nan

            elif pred_empty and not gt_empty:
                cov_p = 0.0
                cov_r = 0.0
                line_f1 = 0.0
                mean_p2l = np.nan

            elif (not pred_empty) and gt_empty:
                cov_p = 0.0
                cov_r = 0.0
                line_f1 = 0.0
                mean_p2l = np.nan

            else:
                cov_p = float(np.mean(dist_P2L <= thresh)) if len(dist_P2L) > 0 else 0.0
                cov_r = float(np.mean(dist_G2P <= thresh)) if len(dist_G2P) > 0 else 0.0
                line_f1 = 2 * cov_p * cov_r / (cov_p + cov_r) if (cov_p + cov_r) > 0 else 0.0
                mean_p2l = float(np.mean(dist_P2L)) if len(dist_P2L) > 0 else np.nan

            scene_res[thresh] = {
                "EGR": cov_r * 100.0,
                "EGP": cov_p * 100.0,
                "GF1": line_f1 * 100.0,
                "EGD": mean_p2l,
                "CR": corner_metrics["CR"] * 100.0,
                "CD": corner_metrics["CD"],
                "LN": float(num_lines),
            }

        return scene_res

    def evaluate_and_accumulate(self, pred_lines, gt_lines):

        scene_res = self.evaluate_scene(pred_lines, gt_lines)

        for thresh in self.thresholds:
            for k, v in scene_res[thresh].items():
                if k not in self.acc_metrics[thresh]:
                    self.acc_metrics[thresh][k] = 0.0
                if k not in self.valid_counts[thresh]:
                    self.valid_counts[thresh][k] = 0

                if is_nan_scalar(v):
                    continue

                self.acc_metrics[thresh][k] += float(v)
                self.valid_counts[thresh][k] += 1

        self.num_samples += 1
        return scene_res

    def get_summary_dict(self, thresh):

        if self.num_samples == 0:
            return {}

        result = {}

        for k in self.acc_metrics[thresh].keys():
            cnt = self.valid_counts[thresh].get(k, 0)
            if cnt > 0:
                result[k] = self.acc_metrics[thresh][k] / cnt
            else:
                result[k] = np.nan

        return result


def parse_prediction(value):
    name, separator, filename = value.partition("=")
    if not separator or not name.strip() or not filename.strip():
        raise argparse.ArgumentTypeError("prediction must use NAME=FILENAME")
    return name.strip(), filename.strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the Point2Contour metrics reported in the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-dir", required=True, help="Directory containing scene subdirectories")
    parser.add_argument("--output-dir", default=None, help="Directory for reports and CSV files")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2], help="Evaluation distance thresholds in meters")
    parser.add_argument("--density", type=float, default=0.1, help="Line sampling interval in meters")
    parser.add_argument("--gt-filename", default="gt_wire.obj", help="Ground-truth filename in each scene")
    parser.add_argument("--prediction", type=parse_prediction, action="append", default=None, metavar="NAME=FILENAME", help="Prediction target; repeat for multiple files")
    args = parser.parse_args()
    if any(threshold <= 0 for threshold in args.thresholds):
        parser.error("thresholds must be positive")
    if args.density <= 0:
        parser.error("density must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    TARGET_DIR = os.path.abspath(os.path.expanduser(args.target_dir))
    OUTPUT_DIR = (
        os.path.abspath(os.path.expanduser(args.output_dir))
        if args.output_dir
        else TARGET_DIR
    )
    EVAL_THRESHOLDS = args.thresholds
    GT_FILENAME = args.gt_filename
    PRED_FILENAMES_TO_EVAL = dict(args.prediction or [
        ("pre_seg", "pre_seg.obj"),
        ("pre_seg_nms", "pre_seg_nms.obj"),
    ])

    if not os.path.exists(TARGET_DIR):
        print(f"Target directory not found: {TARGET_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(OUTPUT_DIR, f"eval_report_{timestamp}.txt")
    sys.stdout = Logger(log_file_path)

    print("Starting evaluation.")
    print(f"Data directory: {TARGET_DIR}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    evaluators = {}
    for eval_name in PRED_FILENAMES_TO_EVAL.keys():

        evaluators[eval_name] = LineCloudEvaluator(
            name=eval_name,
            thresholds=EVAL_THRESHOLDS,
            density=args.density,
        )

    scene_dirs = sorted(
        os.path.join(TARGET_DIR, d)
        for d in os.listdir(TARGET_DIR)
        if os.path.isdir(os.path.join(TARGET_DIR, d))
    )

    if len(scene_dirs) == 0:
        print("No scene directories found.")
        sys.exit(0)

    csv_data_by_thresh = {thr: [] for thr in EVAL_THRESHOLDS}

    missing_pred_files = defaultdict(int)

    pbar = tqdm(scene_dirs, desc="Evaluation")
    valid_scenes = 0

    for scene_dir in pbar:
        scene_name = os.path.basename(scene_dir)
        gt_path = os.path.join(scene_dir, GT_FILENAME)

        if not os.path.exists(gt_path):
            continue

        lines_gt = read_lines_from_obj(gt_path)
        valid_scenes += 1

        for eval_name, pred_filename in PRED_FILENAMES_TO_EVAL.items():
            pred_path = os.path.join(scene_dir, pred_filename)

            if not os.path.exists(pred_path):
                missing_pred_files[eval_name] += 1
                continue

            lines_pred = read_lines_from_obj(pred_path)
            scene_metrics = evaluators[eval_name].evaluate_and_accumulate(lines_pred, lines_gt)

            for thresh in EVAL_THRESHOLDS:
                res = scene_metrics[thresh]

                csv_data_by_thresh[thresh].append([
                    scene_name,
                    eval_name,
                    fmt_float(res["EGR"], 2),
                    fmt_float(res["EGP"], 2),
                    fmt_float(res["GF1"], 2),
                    fmt_float(res["EGD"], 4),
                    fmt_float(res["CR"], 2),
                    fmt_float(res["CD"], 4),
                    int(res["LN"]),
                ])

    print(f"Evaluation completed: {valid_scenes} scenes.")

    if missing_pred_files:
        print("Missing prediction files:")
        for eval_name, cnt in missing_pred_files.items():
            print(f"  {eval_name}: {cnt}")

    csv_header = [
        "Scene_Name", "Target_Name",
        "EGR(%)", "EGP(%)", "GF1(%)", "EGD(m)", "CR(%)", "CD(m)", "LN",
    ]

    for TARGET_THRESH in EVAL_THRESHOLDS:
        print(f"\nThreshold: {TARGET_THRESH} m")

        cols = [
            ("Target", 24),
            ("EGR (%)", 10),
            ("EGP (%)", 10),
            ("GF1 (%)", 10),
            ("EGD (m)", 10),
            ("CR (%)", 10),
            ("CD (m)", 10),
            ("LN", 8),
        ]

        header = " | ".join(pad_str(column, width) for column, width in cols)
        print(header)
        print("-" * len(header))

        for eval_name, evaluator in evaluators.items():
            res = evaluator.get_summary_dict(TARGET_THRESH)
            if not res:
                continue

            row_vals = [
                pad_str(eval_name, cols[0][1]),
                pad_str(fmt_float(res["EGR"], 2), cols[1][1]),
                pad_str(fmt_float(res["EGP"], 2), cols[2][1]),
                pad_str(fmt_float(res["GF1"], 2), cols[3][1]),
                pad_str(fmt_float(res["EGD"], 4), cols[4][1]),
                pad_str(fmt_float(res["CR"], 2), cols[5][1]),
                pad_str(fmt_float(res["CD"], 4), cols[6][1]),
                pad_str(fmt_float(res["LN"], 2), cols[7][1]),
            ]

            print(" | ".join(row_vals))

        csv_file_path = os.path.join(OUTPUT_DIR, f"eval_scene_metrics_{TARGET_THRESH}m_{timestamp}.csv")

        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(csv_data_by_thresh[TARGET_THRESH])

        print(f"Scene metrics: {os.path.basename(csv_file_path)}")

    print(f"Report: {log_file_path}")
