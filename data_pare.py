import argparse
import os
import sys
import math
import pickle
import logging
import random
import re
from datetime import datetime

import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
SPLIT_SEED = 42
_NUMERIC_PART = re.compile(r"(\d+)")


def as_points_array(arr, name="array"):

    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.atleast_2d(arr)
    if arr.shape[1] != 3:
        raise ValueError(f"{name} should have shape [N, 3], got {arr.shape}")
    return arr


def build_csr_topology(corner_link, num_corners):

    num_corners = int(num_corners)
    adj = [[] for _ in range(num_corners)]

    for u, v in np.asarray(corner_link, dtype=np.int64).reshape(-1, 2):
        u, v = int(u), int(v)
        if 0 <= u < num_corners and 0 <= v < num_corners and u != v:
            adj[u].append(v)
            adj[v].append(u)

    indptr = np.zeros(num_corners + 1, dtype=np.int64)
    indices = []
    for i in range(num_corners):
        indices.extend(sorted(set(adj[i])))
        indptr[i + 1] = len(indices)

    return indptr, np.asarray(indices, dtype=np.int64)


def save_xyz_file(path, data, fmt="%.8f"):
    if data is None or data.shape[0] == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, data, fmt=fmt)


def fps_downsample_indices(points, target_n, seed=42):

    points = as_points_array(points, name="points")
    N = points.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.int64)

    target_n = min(int(target_n), N)
    target_n = max(target_n, 1)

    sel = np.empty(target_n, dtype=np.int64)
    points_64 = points.astype(np.float64, copy=False)

    centroid = points_64.mean(axis=0)
    d2_cent = np.sum((points_64 - centroid) ** 2, axis=1)
    sel[0] = int(np.argmax(d2_cent))

    last = points_64[sel[0]]
    min_d2 = np.full(N, np.inf, dtype=np.float64)

    for i in range(1, target_n):
        diff = points_64 - last
        cur_d2 = np.einsum("ij,ij->i", diff, diff)
        min_d2 = np.minimum(min_d2, cur_d2)
        sel[i] = int(np.argmax(min_d2))
        last = points_64[sel[i]]

    return sel


def _safe_query_knn(query_xyz, support_xyz, k):

    query_xyz = as_points_array(query_xyz, name="query_xyz")
    support_xyz = as_points_array(support_xyz, name="support_xyz")

    Nq = query_xyz.shape[0]
    Ns = support_xyz.shape[0]
    k = int(k)

    if Ns == 0:
        raise ValueError("support_xyz is empty; cannot query kNN.")
    if Nq == 0:
        return (
            np.zeros((0, k), dtype=np.int64),
            np.zeros((0, k), dtype=np.float32),
        )

    k_eff = min(k, Ns)
    tree = cKDTree(support_xyz)
    dists, idx = tree.query(query_xyz, k=k_eff)

    if k_eff == 1:
        dists = dists.reshape(Nq, 1)
        idx = idx.reshape(Nq, 1)

    if k_eff < k:
        pad_n = k - k_eff
        idx_pad = np.repeat(idx[:, :1], pad_n, axis=1)
        dist_pad = np.repeat(dists[:, :1], pad_n, axis=1)
        idx = np.concatenate([idx, idx_pad], axis=1)
        dists = np.concatenate([dists, dist_pad], axis=1)

    return idx.astype(np.int64, copy=False), dists.astype(np.float32, copy=False)


def build_token_receptive_fields(
    points,
    B=512,
    K=64,
    K_large=128,
    radius=0.16,
    seed=0,
    strategy="knn",
    dilation_step=2,
):
    points = as_points_array(points, name="points")
    N = int(points.shape[0])
    if N == 0:
        raise ValueError("points is empty; cannot build token receptive fields.")

    B_eff = int(min(B, N))
    K = int(max(1, K))
    K_large = int(max(K_large, K))

    if strategy == "adaptive_dilated":
        K_large = max(K_large, K * int(max(1, dilation_step)))

    K_large_eff = min(K_large, N)

    centers_idx = fps_downsample_indices(points, B_eff, seed=seed)
    centers_xyz = points[centers_idx]

    idx_large, dist_large = _safe_query_knn(centers_xyz, points, k=K_large_eff)

    if strategy == "knn":
        group_idx = idx_large[:, :min(K, idx_large.shape[1])]
        group_dist = dist_large[:, :min(K, dist_large.shape[1])]

    elif strategy == "adaptive_dilated":
        step = int(max(1, dilation_step))
        d64_col = min(K - 1, dist_large.shape[1] - 1)
        d64 = dist_large[:, d64_col]
        dense_threshold = float(np.median(d64))
        dense_mask = d64 <= dense_threshold

        normal_idx = idx_large[:, :min(K, idx_large.shape[1])]
        normal_dist = dist_large[:, :min(K, dist_large.shape[1])]

        dilated_cols = np.arange(0, min(K * step, idx_large.shape[1]), step, dtype=np.int64)
        if dilated_cols.shape[0] < K:
            extra_cols = np.arange(idx_large.shape[1], dtype=np.int64)
            dilated_cols = np.unique(np.concatenate([dilated_cols, extra_cols]))[:K]
        else:
            dilated_cols = dilated_cols[:K]

        dilated_idx = idx_large[:, dilated_cols]
        dilated_dist = dist_large[:, dilated_cols]

        group_idx = normal_idx.copy()
        group_dist = normal_dist.copy()
        group_idx[dense_mask] = dilated_idx[dense_mask]
        group_dist[dense_mask] = dilated_dist[dense_mask]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if group_idx.shape[1] < K:
        pad_n = K - group_idx.shape[1]
        idx_pad = np.repeat(group_idx[:, :1], pad_n, axis=1)
        dist_pad = np.repeat(group_dist[:, :1], pad_n, axis=1)
        group_idx = np.concatenate([group_idx, idx_pad], axis=1)
        group_dist = np.concatenate([group_dist, dist_pad], axis=1)

    return (
        group_idx.astype(np.int64, copy=False),
        centers_idx.astype(np.int64, copy=False),
    )


def compute_fp_indices(points, centers_xyz, k=3):

    idx, dists = _safe_query_knn(points, centers_xyz, k=int(k))
    return idx.astype(np.int64, copy=False), dists.astype(np.float32, copy=False)


def _point_to_segment_distances(P, A, B):
    P = P.astype(np.float64, copy=False)
    A = A.astype(np.float64, copy=False)
    B = B.astype(np.float64, copy=False)

    AB = B - A
    AP = P - A
    ab2 = float(np.dot(AB, AB))
    if ab2 <= 1e-18:
        return np.linalg.norm(AP, axis=1)

    t = np.einsum("ij,j->i", AP, AB) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = A[None, :] + t[:, None] * AB[None, :]
    return np.linalg.norm(P - proj, axis=1)


def _edges_as_pairs(edges):
    pairs = []
    for e in edges:
        if len(e) < 2:
            continue
        for a, b in zip(e[:-1], e[1:]):
            pairs.append((int(a), int(b)))
    return pairs


def natural_sort_key(value):
    parts = _NUMERIC_PART.split(str(value))
    key = []
    for index, part in enumerate(parts):
        if not part:
            continue
        key.append((1, int(part)) if index % 2 == 1 else (0, part))
    return key


def index_scene_files(root_path, extension):
    id2path = {}
    extension = extension.lower()
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(extension):
                scene_id = os.path.splitext(file)[0]
                id2path[scene_id] = os.path.join(root, file)
    return id2path


def split_scene_ids(scene_ids):
    all_ids = sorted(set(scene_ids), key=natural_sort_key)
    shuffled_ids = all_ids.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled_ids)
    train_count = int(round(len(shuffled_ids) * TRAIN_RATIO))
    train_ids = shuffled_ids[:train_count]
    val_ids = shuffled_ids[train_count:]
    return all_ids, train_ids, val_ids


def write_id_list(path, scene_ids):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(scene_ids))


def KNN_idx(pc, leafsize, include_self=True):
    pc = as_points_array(pc, name="pc")
    N = pc.shape[0]
    if N == 0:
        return np.zeros((0, int(leafsize)), dtype=np.int64)

    leafsize = int(max(1, leafsize))
    query_k = leafsize if include_self else leafsize + 1
    idx, _ = _safe_query_knn(pc, pc, k=query_k)

    if include_self:
        out = idx[:, :leafsize]
    else:
        out = idx[:, 1:leafsize + 1]
        if out.shape[1] < leafsize:
            pad_n = leafsize - out.shape[1]
            out = np.concatenate([out, np.repeat(idx[:, :1], pad_n, axis=1)], axis=1)

    return out.astype(np.int64, copy=False)


def _local_density_scale(pc, k=16, use_median=True):
    pc = as_points_array(pc, name="pc")
    N = pc.shape[0]
    if N <= 1:
        return np.ones((N,), dtype=np.float64)

    k_eff = min(int(k) + 1, N)
    tree = cKDTree(pc)
    dist, _ = tree.query(pc, k=k_eff)

    if k_eff == 1:
        rho = np.ones((N,), dtype=np.float64)
    else:
        vals = dist[:, 1:]
        rho = np.median(vals, axis=1) if use_median else np.mean(vals, axis=1)

    lo, hi = np.percentile(rho, [5, 95])
    lo = max(float(lo), 1e-9)
    hi = max(float(hi), lo)
    return np.clip(rho, lo, hi).astype(np.float64, copy=False)


def compute_density_adaptive_soft_labels(
    pc_full,
    vertices_norm,
    edges_list,
    k_density=16,
    kappa=2.0,
    dmax_percentile=95.0,
):
    pc_full = as_points_array(pc_full, name="pc_full")
    V = as_points_array(vertices_norm, name="vertices_norm")
    N = int(pc_full.shape[0])
    pairs = _edges_as_pairs(edges_list)

    if len(pairs) == 0 or V.shape[0] == 0:
        zeros = np.zeros(N, dtype=np.float32)
        ones = np.ones(N, dtype=np.float32)
        return {
            "edge_soft_label": zeros,
            "edge_dist": ones,
            "edge_density_rho": ones,
            "edge_sigma": ones,
        }

    d_min = np.full(N, np.inf, dtype=np.float64)
    for vi, vj in pairs:
        if 0 <= vi < V.shape[0] and 0 <= vj < V.shape[0]:
            d_edge = _point_to_segment_distances(pc_full, V[vi], V[vj])
            d_min = np.minimum(d_min, d_edge)

    finite = np.isfinite(d_min)
    if finite.any():
        dmax = float(np.percentile(d_min[finite], dmax_percentile))
    else:
        dmax = 1.0
    dmax = max(dmax, 1e-9)

    rho = _local_density_scale(pc_full, k=k_density)
    sigma = np.maximum(float(kappa) * rho, 1e-9)
    y = np.exp(-((np.clip(d_min, 0.0, dmax) / sigma) ** 2))

    return {
        "edge_soft_label": y.astype(np.float32, copy=False),
        "edge_dist": d_min.astype(np.float32, copy=False),
        "edge_density_rho": rho.astype(np.float32, copy=False),
        "edge_sigma": sigma.astype(np.float32, copy=False),
    }


def generate_base_rays(n_ele=6, n_azi=12):
    vectors = []
    azi_step = (2 * math.pi) / int(n_azi)
    ele_step = math.pi / int(n_ele)
    for ele_idx in range(int(n_ele)):
        theta = ele_idx * ele_step + ele_step / 2.0
        for azi_idx in range(int(n_azi)):
            phi = azi_idx * azi_step + azi_step / 2.0
            vectors.append([
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ])
    return np.asarray(vectors, dtype=np.float32)


def extract_corner_links(edges_list):
    edge_pairs = _edges_as_pairs(edges_list)
    unique_links = {tuple(sorted((int(vi), int(vj)))) for vi, vj in edge_pairs if vi != vj}
    return np.asarray(list(unique_links), dtype=np.int64) if unique_links else np.zeros((0, 2), dtype=np.int64)


def normalize_point_data(data_dict):
    points = as_points_array(data_dict["points"], name="points")
    vertices = as_points_array(data_dict.get("vertices", np.zeros((0, 3))), name="vertices")

    if points.shape[0] == 0 and vertices.shape[0] == 0:
        raise ValueError("Both points and vertices are empty.")

    if points.shape[0] > 0 and vertices.shape[0] > 0:
        all_pts = np.concatenate([points, vertices], axis=0)
    elif points.shape[0] > 0:
        all_pts = points
    else:
        all_pts = vertices

    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = ((mins + maxs) / 2.0).astype(np.float64)

    scale = float(np.max(maxs - mins) / 2.0)
    factor = float(0.9 / (scale if scale > 1e-12 else 1.0))

    data_dict["points"] = (points - center) * factor
    data_dict["vertices"] = (vertices - center) * factor
    data_dict["center"] = center
    data_dict["factor"] = factor
    return data_dict


def load_xyz_points(xyz_path):
    points = np.loadtxt(xyz_path, usecols=(0, 1, 2)).astype(np.float64)
    points = np.atleast_2d(points)
    if points.shape[1] != 3:
        raise ValueError(f"xyz should have 3 columns, got shape={points.shape}")
    return points


def load_wireframe_obj(obj_path):
    vertices, edges = [], []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append(list(map(float, parts[1:4])))
            elif parts[0] == "l" and len(parts) >= 3:
                edges.append([int(x.split("/")[0]) - 1 for x in parts[1:]])

    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    return vertices, edges


def main(dataset_root, output_path):
    train_root = os.path.join(dataset_root, "train")
    xyz_root = os.path.join(train_root, "xyz")
    wireframe_root = os.path.join(train_root, "wireframe")
    if not os.path.isdir(xyz_root):
        raise FileNotFoundError(f"XYZ directory not found: {xyz_root}")
    if not os.path.isdir(wireframe_root):
        raise FileNotFoundError(f"Wireframe directory not found: {wireframe_root}")

    id2xyz = index_scene_files(xyz_root, ".xyz")
    id2obj = index_scene_files(wireframe_root, ".obj")
    xyz_ids = set(id2xyz)
    obj_ids = set(id2obj)
    missing_xyz = sorted(obj_ids - xyz_ids, key=natural_sort_key)
    missing_obj = sorted(xyz_ids - obj_ids, key=natural_sort_key)
    if missing_xyz or missing_obj:
        details = []
        if missing_xyz:
            details.append(f"missing XYZ for {len(missing_xyz)} IDs: {missing_xyz[:5]}")
        if missing_obj:
            details.append(f"missing OBJ for {len(missing_obj)} IDs: {missing_obj[:5]}")
        raise FileNotFoundError("Input pairs are incomplete; " + "; ".join(details))

    scene_ids = sorted(xyz_ids, key=natural_sort_key)
    if not scene_ids:
        raise FileNotFoundError(f"No paired XYZ/OBJ scenes found under: {train_root}")

    point_cloud_config = {
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
    density_neighbors = 16
    soft_label_kappa = 0.70 / max(math.sqrt(-math.log(0.50)), 1e-8)
    distance_percentile = 95.0
    base_rays = generate_base_rays(n_ele=6, n_azi=12)
    successful_ids = []
    failed_ids = []
    os.makedirs(output_path, exist_ok=True)

    logging.info(f"Found {len(scene_ids)} paired training scenes.")
    with tqdm(total=len(scene_ids), desc="Converting", unit="scene", dynamic_ncols=True) as pbar:
        for scene_id in scene_ids:
            try:
                xyz_path = id2xyz[scene_id]
                obj_path = id2obj[scene_id]
                pc_raw_initial = load_xyz_points(xyz_path)

                if pc_raw_initial.shape[0] == 0:
                    raise ValueError("Empty point cloud")

                vertices, edges = load_wireframe_obj(obj_path)

                target_out = os.path.join(output_path, scene_id)
                os.makedirs(target_out, exist_ok=True)

                norm_data_dict = normalize_point_data({
                    "points": pc_raw_initial,
                    "vertices": vertices,
                    "edges": edges,
                })

                pc_full = as_points_array(norm_data_dict["points"], name="pc_full")
                corner = as_points_array(norm_data_dict["vertices"], name="corner")
                corner_link = extract_corner_links(norm_data_dict["edges"])

                num_corners = corner.shape[0]
                indptr, indices = build_csr_topology(corner_link, num_corners)

                soft_pack = compute_density_adaptive_soft_labels(
                    pc_full,
                    corner,
                    norm_data_dict["edges"],
                    k_density=density_neighbors,
                    kappa=soft_label_kappa,
                    dmax_percentile=distance_percentile,
                )

                pc_KNN_idx_f = KNN_idx(
                    pc_full,
                    leafsize=point_cloud_config["N_knn"],
                    include_self=point_cloud_config["include_self_in_knn"],
                )

                (
                    block_idx_f,
                    centers_idx_f,
                ) = build_token_receptive_fields(
                    pc_full,
                    B=point_cloud_config["block_B"],
                    K=point_cloud_config["block_K"],
                    K_large=point_cloud_config["block_K_large"],
                    radius=point_cloud_config["block_radius"],
                    strategy=point_cloud_config["block_strategy"],
                    dilation_step=point_cloud_config["dilation_step"],
                )

                centers_xyz_f = pc_full[centers_idx_f]
                fp_idx_f, fp_dist_f = compute_fp_indices(
                    pc_full,
                    centers_xyz_f,
                    k=point_cloud_config["fp_k"],
                )

                extra_out_full = {
                    "pc": pc_full.astype(np.float32, copy=False),
                    "n_factor": float(norm_data_dict["factor"]),
                    "n_center": norm_data_dict["center"].astype(np.float64, copy=False),

                    "corner_vertices_xyz": corner.astype(np.float32, copy=False),
                    "corner_link": corner_link.astype(np.int64, copy=False),
                    "corner_adj_indptr": indptr.astype(np.int64, copy=False),
                    "corner_adj_indices": indices.astype(np.int64, copy=False),

                    "pc_KNN_idx": pc_KNN_idx_f.astype(np.int64, copy=False),

                    "block_idx": block_idx_f.astype(np.int64, copy=False),
                    "centers_idx": centers_idx_f.astype(np.int64, copy=False),
                    "fp_idx": fp_idx_f.astype(np.int64, copy=False),
                    "fp_dist": fp_dist_f.astype(np.float32, copy=False),

                    "edge_soft_label": soft_pack["edge_soft_label"].astype(np.float32, copy=False),
                    "edge_dist": soft_pack["edge_dist"].astype(np.float32, copy=False),
                    "edge_density_rho": soft_pack["edge_density_rho"].astype(np.float32, copy=False),
                    "edge_sigma": soft_pack["edge_sigma"].astype(np.float32, copy=False),

                    "spherical_basedir": base_rays.astype(np.float32, copy=False),

                    "point_cloud_config": dict(point_cloud_config),
                }

                with open(os.path.join(target_out, "pc_with_edge_c_full.pkl"), "wb") as f:
                    pickle.dump(extra_out_full, f, protocol=pickle.HIGHEST_PROTOCOL)

                successful_ids.append(scene_id)
                pbar.set_postfix_str(
                    f"id={scene_id}, points={pc_full.shape[0]}, tokens={centers_idx_f.shape[0]}"
                )

            except Exception as e:
                failed_ids.append(scene_id)
                logging.exception(f"Failed {scene_id}: {e}")
            finally:
                pbar.update(1)

    if not successful_ids:
        raise RuntimeError("No scenes were processed successfully.")

    all_ids, train_ids, val_ids = split_scene_ids(successful_ids)
    write_id_list(os.path.join(output_path, "all.txt"), all_ids)
    write_id_list(os.path.join(output_path, "train.txt"), train_ids)
    write_id_list(os.path.join(output_path, "val.txt"), val_ids)

    logging.info(
        f"Data conversion completed: all={len(all_ids)}, train={len(train_ids)}, "
        f"val={len(val_ids)}, failed={len(failed_ids)}."
    )
    logging.info(
        f"Split policy: train={TRAIN_RATIO:.1f}, val={VAL_RATIO:.1f}, seed={SPLIT_SEED}."
    )
    return {
        "all_ids": all_ids,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "failed_ids": failed_ids,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Point2Contour training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root containing train/xyz and train/wireframe",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for processed PKL files and split lists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = os.path.abspath(os.path.expanduser(args.data_root))
    output_path = os.path.abspath(os.path.expanduser(args.output_dir))
    log_dir = os.path.join(output_path, "logs")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"preprocess_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.info(f"Dataset root: {dataset_root}")
    logging.info(f"Output directory: {output_path}")
    main(dataset_root, output_path)
