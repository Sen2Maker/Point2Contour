from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import os.path as op
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm


ROOT_PATH = Path(__file__).resolve().parent


XYZ_ROOT: Optional[Path] = None


GT_OBJ_ROOT: Optional[Path] = None
GT_OBJ_FILE_PATTERN = "{id}.obj"


MODEL_LOG_PATH: Optional[Path] = None
CHECKPOINT = "final"


OUTPUT_PARENT = ROOT_PATH / "res_pre"


ID_SOURCE: Optional[Union[str, Sequence[str]]] = None


MAX_FILES: Optional[int] = None


SKIP_EXISTING = False
FINAL_OUTPUT_NAME = "pre_seg_nms.obj"
RAW_TOPK_OUTPUT_NAME = "raw_topk.obj"


CONTINUE_ON_ERROR = True


REFINE_THR = 0.75
STAGE_THR = 0.75
NUM_ITERS = 3
MIN_LEN = 0.15


POINT_CLOUD_CONFIG = {
    "block_B": 512,
    "block_K": 64,
    "block_K_large": 128,
    "block_radius": 0.16,
    "block_strategy": "knn",
    "dilation_step": 2,
    "fp_k": 3,
    "N_knn": 8,
    "include_self_in_knn": True,
}

NMS_CONFIG = {
    "corner_merge_dist": 0.12,
    "angle_nms_deg": 10.0,
    "max_rays_per_corner": 3,
    "strong_thr": 0.75,
    "top1_thr": 0.85,
    "top2_sum_thr": 1.50,
    "min_strong_rays": 2,
}


def load_py_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

pre2obj = None
prep = load_py_module("xyz_preprocess_utils", ROOT_PATH / "data_pare.py")


def ensure_pre2obj_loaded():
    global pre2obj
    if pre2obj is None:
        pre2obj = load_py_module("pre2obj_utils", ROOT_PATH / "pre.py")
    return pre2obj


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Point2Contour inference on raw XYZ files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xyz-root", required=True, help="XYZ file or directory containing XYZ files")
    parser.add_argument("--gt-root", default=None, help="Optional directory containing GT OBJ files")
    parser.add_argument("--gt-pattern", default=GT_OBJ_FILE_PATTERN, help="GT filename pattern with {id}")
    parser.add_argument("--model-dir", required=True, help="Training result directory")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="Checkpoint name: final or an epoch number")
    parser.add_argument("--output-dir", default=None, help="Exact output directory; defaults to res_pre/<dataset>_<timestamp>")
    parser.add_argument("--ids", default=None, help="ID list, comma-separated IDs, or a text file")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of XYZ files")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Inference device")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenes with the final output file")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed scene")
    parser.add_argument("--refine-threshold", type=float, default=REFINE_THR, help="Line refinement probability threshold")
    parser.add_argument("--stage-threshold", type=float, default=STAGE_THR, help="Stage output probability threshold")
    parser.add_argument("--num-iters", type=int, default=NUM_ITERS, help="Bidirectional refinement rounds")
    parser.add_argument("--min-line-length", type=float, default=MIN_LEN, help="Minimum output line length in meters")
    parser.add_argument("--block-count", type=int, default=POINT_CLOUD_CONFIG["block_B"], help="Maximum token centers")
    parser.add_argument("--block-size", type=int, default=POINT_CLOUD_CONFIG["block_K"], help="Points per token block")
    parser.add_argument("--large-block-size", type=int, default=POINT_CLOUD_CONFIG["block_K_large"], help="Candidates per token block")
    parser.add_argument("--block-radius", type=float, default=POINT_CLOUD_CONFIG["block_radius"], help="Token block radius")
    parser.add_argument("--block-strategy", choices=("knn", "adaptive_dilated"), default=POINT_CLOUD_CONFIG["block_strategy"], help="Token neighborhood strategy")
    parser.add_argument("--dilation-step", type=int, default=POINT_CLOUD_CONFIG["dilation_step"], help="Adaptive dilation step")
    parser.add_argument("--fp-k", type=int, default=POINT_CLOUD_CONFIG["fp_k"], help="Feature propagation neighbors")
    parser.add_argument("--knn", type=int, default=POINT_CLOUD_CONFIG["N_knn"], help="Point neighborhood size")
    parser.add_argument("--exclude-self-in-knn", action="store_true", help="Exclude each query point from its neighborhood")
    parser.add_argument("--corner-merge-distance", type=float, default=NMS_CONFIG["corner_merge_dist"], help="NMS corner merge distance in meters")
    parser.add_argument("--angle-nms-deg", type=float, default=NMS_CONFIG["angle_nms_deg"], help="NMS direction threshold in degrees")
    parser.add_argument("--max-rays-per-corner", type=int, default=NMS_CONFIG["max_rays_per_corner"], help="Maximum retained rays per corner")
    parser.add_argument("--strong-ray-threshold", type=float, default=NMS_CONFIG["strong_thr"], help="Strong ray score threshold")
    parser.add_argument("--top1-threshold", type=float, default=NMS_CONFIG["top1_thr"], help="Single-ray corner acceptance threshold")
    parser.add_argument("--top2-sum-threshold", type=float, default=NMS_CONFIG["top2_sum_thr"], help="Two-ray corner acceptance threshold")
    parser.add_argument("--min-strong-rays", type=int, default=NMS_CONFIG["min_strong_rays"], help="Strong rays required to retain a corner")
    args = parser.parse_args()
    if args.num_iters < 1:
        parser.error("num-iters must be positive")
    return args


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_name)


def infer_dataset_name(input_path):
    candidate = Path(input_path).expanduser().resolve()
    if candidate.is_file() or candidate.suffix.lower() == ".xyz":
        return candidate.stem or "point_cloud"
    if candidate.name.lower() in {"xyz", "points", "pointcloud", "point_cloud"}:
        candidate = candidate.parent
    if candidate.name.lower() in {"train", "val", "test"}:
        candidate = candidate.parent
    return candidate.name or "dataset"


def parse_id_source(id_source: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if id_source is None:
        return None

    if isinstance(id_source, (list, tuple, set)):
        raw_items = [str(x) for x in id_source]
    else:
        text = str(id_source).strip()
        possible_path = Path(text).expanduser()
        if possible_path.is_file():
            text = possible_path.read_text(encoding="utf-8", errors="ignore")
        raw_items = (
            text.replace(",", " ")
            .replace(";", " ")
            .replace("\n", " ")
            .replace("\t", " ")
            .split(" ")
        )

    ids: List[str] = []
    seen = set()
    for item in raw_items:
        sid = Path(str(item).strip()).name
        if sid.lower().endswith(".xyz"):
            sid = sid[:-4]
        if sid and sid not in seen:
            ids.append(sid)
            seen.add(sid)
    return ids


def collect_xyz_files(xyz_root: Path, id_source: Optional[Union[str, Sequence[str]]]) -> List[Path]:
    xyz_root = Path(xyz_root).expanduser().resolve()
    wanted_ids = parse_id_source(id_source)

    if xyz_root.is_file():
        if xyz_root.suffix.lower() != ".xyz":
            raise ValueError(f"Expected an XYZ file: {xyz_root}")
        files = [xyz_root]
        if wanted_ids is not None and xyz_root.stem not in wanted_ids:
            files = []
    elif not xyz_root.exists():
        raise FileNotFoundError(f"XYZ input not found: {xyz_root}")
    elif not xyz_root.is_dir():
        raise ValueError(f"XYZ input must be a file or directory: {xyz_root}")
    elif wanted_ids is None:
        files = sorted(xyz_root.glob("*.xyz"), key=lambda p: p.stem)
    else:
        files = []
        missing = []
        for sid in wanted_ids:
            p = xyz_root / f"{sid}.xyz"
            if p.is_file():
                files.append(p)
            else:
                missing.append(sid)
        if missing:
            print(f"Missing {len(missing)} XYZ files: {missing[:3]}")

    if MAX_FILES is not None:
        files = files[: int(MAX_FILES)]

    return files


def find_gt_obj(sample_id: str) -> Optional[Path]:
    if GT_OBJ_ROOT is None:
        return None

    gt_root = Path(GT_OBJ_ROOT).expanduser().resolve()
    direct = gt_root / GT_OBJ_FILE_PATTERN.format(id=sample_id)
    if direct.is_file():
        return direct

    nested = gt_root / "wireframe" / GT_OBJ_FILE_PATTERN.format(id=sample_id)
    if nested.is_file():
        return nested

    return None


def build_point_only_pack(xyz_path: Path) -> Dict[str, object]:
    pc_raw_initial = prep.load_xyz_points(str(xyz_path))

    if pc_raw_initial.shape[0] == 0:
        raise ValueError(f"Empty point cloud: {xyz_path}")

    gt_obj_path = find_gt_obj(xyz_path.stem)
    if gt_obj_path is None:
        vertices = np.zeros((0, 3), dtype=np.float64)
        edges = []
    else:
        vertices, edges = prep.load_wireframe_obj(str(gt_obj_path))

    norm_data_dict = prep.normalize_point_data({
        "points": pc_raw_initial,
        "vertices": vertices,
        "edges": edges,
    })

    pc_full = prep.as_points_array(norm_data_dict["points"], name="pc_full")
    corner = prep.as_points_array(norm_data_dict["vertices"], name="corner")
    corner_link = prep.extract_corner_links(norm_data_dict["edges"])
    n_points = pc_full.shape[0]
    indptr, indices = prep.build_csr_topology(corner_link, corner.shape[0])

    pc_KNN_idx_f = prep.KNN_idx(
        pc_full,
        leafsize=POINT_CLOUD_CONFIG["N_knn"],
        include_self=POINT_CLOUD_CONFIG["include_self_in_knn"],
    )

    block_idx_f, centers_idx_f = prep.build_token_receptive_fields(
        pc_full,
        B=POINT_CLOUD_CONFIG["block_B"],
        K=POINT_CLOUD_CONFIG["block_K"],
        K_large=POINT_CLOUD_CONFIG["block_K_large"],
        radius=POINT_CLOUD_CONFIG["block_radius"],
        strategy=POINT_CLOUD_CONFIG["block_strategy"],
        dilation_step=POINT_CLOUD_CONFIG["dilation_step"],
    )

    centers_xyz_f = pc_full[centers_idx_f]
    fp_idx_f, fp_dist_f = prep.compute_fp_indices(
        pc_full,
        centers_xyz_f,
        k=POINT_CLOUD_CONFIG["fp_k"],
    )

    base_rays = prep.generate_base_rays(n_ele=6, n_azi=12)

    return {
        "pc": pc_full.astype(np.float32, copy=False),
        "n_factor": float(norm_data_dict["factor"]),
        "n_center": norm_data_dict["center"].astype(np.float64, copy=False),

        "corner_vertices_xyz": corner.astype(np.float32, copy=False),
        "corner_link": corner_link.astype(np.int64, copy=False),
        "corner_adj_indptr": indptr.astype(np.int64, copy=False),
        "corner_adj_indices": indices.astype(np.int64, copy=False),
        "gt_obj_path": "" if gt_obj_path is None else str(gt_obj_path),

        "pc_KNN_idx": pc_KNN_idx_f.astype(np.int64, copy=False),
        "block_idx": block_idx_f.astype(np.int64, copy=False),
        "centers_idx": centers_idx_f.astype(np.int64, copy=False),
        "fp_idx": fp_idx_f.astype(np.int64, copy=False),
        "fp_dist": fp_dist_f.astype(np.float32, copy=False),

        "edge_soft_label": np.zeros((n_points,), dtype=np.float32),
        "edge_dist": np.ones((n_points,), dtype=np.float32),
        "edge_density_rho": np.ones((n_points,), dtype=np.float32),
        "edge_sigma": np.ones((n_points,), dtype=np.float32),

        "spherical_basedir": base_rays.astype(np.float32, copy=False),
        "point_cloud_config": dict(POINT_CLOUD_CONFIG),
    }


def pack_to_tensors(pack: Dict[str, object]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, object]]:
    pc = pack["pc"]
    edge_soft_label = pack["edge_soft_label"]
    base_rays = pack["spherical_basedir"]
    corner_xyz = pack["corner_vertices_xyz"]
    corner_link = pack["corner_link"]
    corner_adj_indptr = pack["corner_adj_indptr"]
    corner_adj_indices = pack["corner_adj_indices"]
    n_center = pack["n_center"]
    n_factor = pack["n_factor"]

    model_input = {
        "pc": torch.from_numpy(pc).float(),
        "pc_KNN_idx": torch.from_numpy(pack["pc_KNN_idx"].astype(np.int64)),
        "centers_idx": torch.from_numpy(pack["centers_idx"].astype(np.int64)),
        "block_idx": torch.from_numpy(pack["block_idx"].astype(np.int64)),
        "fp_idx": torch.from_numpy(pack["fp_idx"].astype(np.int64)),
        "fp_dist": torch.from_numpy(pack["fp_dist"].astype(np.float32)),
        "edge_soft_label": torch.from_numpy(edge_soft_label.astype(np.float32)),
        "flat_rays": torch.from_numpy(base_rays).float(),


        "corner_xyz": torch.from_numpy(corner_xyz.astype(np.float32)),
        "corner_adj_indptr": torch.from_numpy(corner_adj_indptr.astype(np.int64)),
        "corner_adj_indices": torch.from_numpy(corner_adj_indices.astype(np.int64)),
        "n_center": torch.tensor(n_center, dtype=torch.float64),
        "n_factor": torch.tensor(n_factor, dtype=torch.float64),
        "corner_link": torch.from_numpy(corner_link.astype(np.int64)).long(),
    }

    gt = {
        "edge_soft_label": torch.from_numpy(edge_soft_label.astype(np.float32)),
        "n_center": torch.tensor(n_center, dtype=torch.float64),
        "n_factor": torch.tensor(n_factor, dtype=torch.float64),
        "base_rays": torch.from_numpy(base_rays).float(),
        "corner_link": torch.from_numpy(corner_link.astype(np.int64)).long(),
        "corner_xyz": torch.from_numpy(corner_xyz.astype(np.float32)),
        "corner_adj_indptr": torch.from_numpy(corner_adj_indptr.astype(np.int64)),
        "corner_adj_indices": torch.from_numpy(corner_adj_indices.astype(np.int64)),
    }

    info = {
        "N": int(pc.shape[0]),
        "B": int(pack["block_idx"].shape[0]),
        "K": int(pack["block_idx"].shape[1]),
        "n_center": n_center,
        "n_factor": n_factor,
    }

    return model_input, gt, info


def move_tensor_dict(data: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in data.items()
    }


def run_one_xyz(
    xyz_path: Path,
    model,
    device: torch.device,
    output_root: Path,
) -> Dict[str, object]:
    pre2obj_utils = ensure_pre2obj_loaded()

    name = xyz_path.stem
    scene_dir = output_root / name
    scene_dir.mkdir(parents=True, exist_ok=True)

    final_output = scene_dir / FINAL_OUTPUT_NAME
    if SKIP_EXISTING and final_output.is_file() and (scene_dir / RAW_TOPK_OUTPUT_NAME).is_file():
        return {
            "name": name,
            "status": "skipped_exists",
            "output_dir": str(scene_dir),
            "xyz_path": str(xyz_path),
        }

    pack = build_point_only_pack(xyz_path)

    model_input, gt, info = pack_to_tensors(pack)
    m_input_cuda = move_tensor_dict(model_input, device)
    gt_cuda = move_tensor_dict(gt, device)

    out = model.predict(
        m_input_cuda,
        refine_thresh=REFINE_THR,
        num_iters=NUM_ITERS,
    )

    real_pc = pre2obj_utils.extract_real_pc(m_input_cuda, gt_cuda)
    lines_gt = pre2obj_utils.extract_gt_lines(gt_cuda)
    edge_probability = torch.sigmoid(out["edge_mask"]).detach().cpu().numpy()
    point_output = np.column_stack((real_pc, edge_probability))

    lines_raw_topk = pre2obj_utils.extract_raw_topk_proposals(
        out,
        gt_cuda,
        min_len=0.0,
    )

    lines_pred, final_rays = pre2obj_utils.extract_history_stage_lines(
        out,
        gt_cuda,
        stage_name=f"iter_{NUM_ITERS - 1}_B_to_A",
        score_mode="round",
        stage_thr=STAGE_THR,
        min_len=MIN_LEN,
    )
    lines_pred_nms, _, _ = pre2obj_utils.linecloud_corner_support_nms(
        final_rays, **NMS_CONFIG
    )

    pre2obj_utils.save_lines_to_obj(scene_dir / "gt_wire.obj", lines_gt)
    pre2obj_utils.save_point_cloud_xyz(scene_dir / "pc.xyz", point_output)
    pre2obj_utils.save_lines_to_obj(scene_dir / RAW_TOPK_OUTPUT_NAME, lines_raw_topk)
    pre2obj_utils.save_lines_to_obj(scene_dir / "pre_seg.obj", lines_pred)
    pre2obj_utils.save_lines_to_obj(scene_dir / "pre_seg_nms.obj", lines_pred_nms)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": name,
        "status": "ok",
        "xyz_path": str(xyz_path),
        "gt_obj_path": str(pack.get("gt_obj_path", "")),
        "has_gt_obj": bool(pack.get("gt_obj_path", "")),
        "output_dir": str(scene_dir),
        "N": info["N"],
        "B": info["B"],
        "K": info["K"],
        "gt_lines": len(lines_gt),
        "raw_topk_lines": len(lines_raw_topk),
        "pred_lines": len(lines_pred),
        "pred_nms_lines": len(lines_pred_nms),
    }


def write_summary(rows: List[Dict[str, object]], output_root: Path) -> Path:
    summary_path = output_root / "xyz_folder_inference_summary.csv"

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return summary_path


def main() -> None:
    global XYZ_ROOT, GT_OBJ_ROOT, GT_OBJ_FILE_PATTERN
    global MODEL_LOG_PATH, CHECKPOINT, ID_SOURCE, MAX_FILES
    global SKIP_EXISTING
    global CONTINUE_ON_ERROR, REFINE_THR, STAGE_THR
    global NUM_ITERS, MIN_LEN

    args = parse_args()
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("max-files must be positive.")
    if "{id}" not in args.gt_pattern:
        raise ValueError("gt-pattern must contain {id}.")

    XYZ_ROOT = Path(args.xyz_root)
    GT_OBJ_ROOT = None if args.gt_root is None else Path(args.gt_root)
    GT_OBJ_FILE_PATTERN = args.gt_pattern
    MODEL_LOG_PATH = Path(args.model_dir)
    CHECKPOINT = args.checkpoint
    ID_SOURCE = args.ids
    MAX_FILES = args.max_files
    SKIP_EXISTING = args.skip_existing
    CONTINUE_ON_ERROR = not args.stop_on_error
    REFINE_THR = args.refine_threshold
    STAGE_THR = args.stage_threshold
    NUM_ITERS = args.num_iters
    MIN_LEN = args.min_line_length

    POINT_CLOUD_CONFIG.update({
        "block_B": args.block_count,
        "block_K": args.block_size,
        "block_K_large": args.large_block_size,
        "block_radius": args.block_radius,
        "block_strategy": args.block_strategy,
        "dilation_step": args.dilation_step,
        "fp_k": args.fp_k,
        "N_knn": args.knn,
        "include_self_in_knn": not args.exclude_self_in_knn,
    })
    NMS_CONFIG.update({
        "corner_merge_dist": args.corner_merge_distance,
        "angle_nms_deg": args.angle_nms_deg,
        "max_rays_per_corner": args.max_rays_per_corner,
        "strong_thr": args.strong_ray_threshold,
        "top1_thr": args.top1_threshold,
        "top2_sum_thr": args.top2_sum_threshold,
        "min_strong_rays": args.min_strong_rays,
    })

    device = resolve_device(args.device)
    pre2obj_utils = ensure_pre2obj_loaded()

    if args.output_dir:
        output_root = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = infer_dataset_name(XYZ_ROOT)
        output_root = Path(OUTPUT_PARENT) / f"{dataset_name}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    log_file_path = output_root / "inference_log.txt"
    sys.stdout = pre2obj_utils.Logger(str(log_file_path))

    xyz_files = collect_xyz_files(XYZ_ROOT, ID_SOURCE)

    print("Starting XYZ inference.")
    print(f"XYZ input: {Path(XYZ_ROOT).expanduser().resolve()}")
    print(f"GT directory: {None if GT_OBJ_ROOT is None else Path(GT_OBJ_ROOT).expanduser().resolve()}")
    print(f"Scenes: {len(xyz_files)}")
    print(f"Model directory: {Path(MODEL_LOG_PATH).expanduser().resolve()}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Output directory: {output_root}")
    print(f"Device: {device}")

    model = pre2obj_utils.load_end2end_model(
        str(Path(MODEL_LOG_PATH).expanduser().resolve()),
        device,
        checkpoint=CHECKPOINT,
    )

    rows: List[Dict[str, object]] = []

    with torch.no_grad():
        pbar = tqdm(xyz_files, desc="Inference", unit="scene")
        for xyz_path in pbar:
            name = xyz_path.stem
            try:
                row = run_one_xyz(
                    xyz_path=xyz_path,
                    model=model,
                    device=device,
                    output_root=output_root,
                )
                rows.append(row)

                if row.get("status") == "ok":
                    print(
                        f"{name}: "
                        f"points={row['N']}, "
                        f"gt={row['gt_lines']}, "
                        f"raw_topk={row['raw_topk_lines']}, "
                        f"pred={row['pred_lines']}, "
                        f"pred_nms={row['pred_nms_lines']}"
                    )
                else:
                    print(f"{name}: {row.get('status')}")

            except Exception as exc:
                err_row = {
                    "name": name,
                    "status": "error",
                    "xyz_path": str(xyz_path),
                    "error": repr(exc),
                }
                rows.append(err_row)
                print(f"Failed {name}: {exc!r}", flush=True)
                if not CONTINUE_ON_ERROR:
                    raise

    summary_path = write_summary(rows, output_root)

    ok_count = sum(1 for r in rows if r.get("status") == "ok")
    err_count = sum(1 for r in rows if r.get("status") == "error")
    skip_count = sum(1 for r in rows if str(r.get("status", "")).startswith("skipped"))

    print("Inference completed.")
    print(f"Processed: {ok_count}")
    print(f"Failed: {err_count}")
    print(f"Skipped: {skip_count}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
