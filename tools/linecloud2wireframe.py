import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


@dataclass(frozen=True)
class Scheme:
    name: str
    cluster_eps: float
    attach_eps: float
    min_samples: int
    min_votes: int
    min_graph_len: float
    max_snap: float
    min_consistency: float
    max_degree: int
    angle_sep_deg: float
    edge_support_radius: float
    min_edge_support: int
    score_support_weight: float
    score_vote_weight: float
    score_snap_weight: float
    path_pair_max_len: float = 0.0
    path_support_radius: float = 0.0
    path_min_support: int = 0
    path_min_coverage: float = 0.0
    path_score_weight: float = 0.0
    path_knn: int = 0
    repair_degree: int = 0
    repair_min_score: float = 0.0
    repair_angle_sep_deg: float = 0.0
    output_repeats: int = 1
    center_method: str = "median"


def load_obj_edges(path):
    vertices = []
    edges = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            parts = raw.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "l" and len(parts) >= 3:
                indices = [int(token.split("/")[0]) - 1 for token in parts[1:]]
                edges.extend(zip(indices[:-1], indices[1:]))
    if not vertices or not edges:
        return np.empty((0, 2, 3), dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    out = [
        [vertices[first], vertices[second]]
        for first, second in edges
        if 0 <= first < len(vertices)
        and 0 <= second < len(vertices)
        and first != second
    ]
    if not out:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def save_obj_edges(path, edges_real):
    path.parent.mkdir(parents=True, exist_ok=True)
    edges_real = np.asarray(edges_real, dtype=np.float64).reshape(-1, 2, 3)
    if edges_real.size == 0:
        path.write_text("", encoding="utf-8")
        return 0, 0
    vertices = edges_real.reshape(-1, 3)
    unique, inverse = np.unique(np.round(vertices, 8), axis=0, return_inverse=True)
    edges = np.sort(inverse.reshape(-1, 2), axis=1)
    edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
    with open(path, "w", encoding="utf-8") as f:
        for vertex in unique:
            f.write("v %.8f %.8f %.8f\n" % tuple(vertex))
        for edge in edges:
            f.write("l %d %d\n" % (edge[0] + 1, edge[1] + 1))
    return int(len(unique)), int(len(edges))


def normalize_edges(edges, center, scale):
    edges = np.asarray(edges, dtype=np.float64).reshape(-1, 2, 3)
    if edges.size == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    return (edges - center.reshape(1, 1, 3)) / scale


def denormalize_edges(edges, center, scale):
    edges = np.asarray(edges, dtype=np.float64).reshape(-1, 2, 3)
    if edges.size == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    return edges * scale + center.reshape(1, 1, 3)


def point_segment_stats(points, first, second, radius, bins=8):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return 0, 0.0
    segment = second - first
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        return 0, 0.0
    positions = ((points - first.reshape(1, 3)) @ segment) / denominator
    mask = (positions >= 0.0) & (positions <= 1.0)
    if not np.any(mask):
        return 0, 0.0
    projections = first.reshape(1, 3) + positions[mask, None] * segment.reshape(1, 3)
    distances = np.linalg.norm(points[mask] - projections, axis=1)
    inlier_positions = positions[mask][distances <= radius]
    if len(inlier_positions) == 0:
        return 0, 0.0
    bin_indices = np.clip((inlier_positions * bins).astype(np.int64), 0, bins - 1)
    return int(len(inlier_positions)), len(set(bin_indices.tolist())) / float(bins)


def cluster_centers(points, labels, method="median"):
    centers = []
    for label in sorted(set(labels) - {-1}):
        cluster = points[labels == label]
        centers.append(
            np.mean(cluster, axis=0)
            if method == "mean"
            else np.median(cluster, axis=0)
        )
    if not centers:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(centers, dtype=np.float64)


def linecloud_to_wireframe(lines, edge_points, scheme):
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 2, 3)
    edge_points = np.asarray(edge_points, dtype=np.float64).reshape(-1, 3)
    if lines.size == 0:
        return np.empty((0, 2, 3), dtype=np.float64), {
            "num_clusters": 0,
            "num_candidates": 0,
            "num_chosen": 0,
        }

    vectors = lines[:, 1] - lines[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths >= scheme.min_graph_len
    lines = lines[valid]
    vectors = vectors[valid]
    lengths = lengths[valid]
    if len(lines) == 0:
        return np.empty((0, 2, 3), dtype=np.float64), {
            "num_clusters": 0,
            "num_candidates": 0,
            "num_chosen": 0,
        }

    endpoints = lines.reshape(-1, 3)
    labels = DBSCAN(
        eps=scheme.cluster_eps,
        min_samples=scheme.min_samples,
    ).fit_predict(endpoints)
    centers = cluster_centers(endpoints, labels, scheme.center_method)
    if len(centers) < 2:
        return np.empty((0, 2, 3), dtype=np.float64), {
            "num_clusters": int(len(centers)),
            "num_candidates": 0,
            "num_chosen": 0,
        }

    distances_to_centers = cdist(endpoints, centers)
    nearest = distances_to_centers.argmin(axis=1)
    nearest_distance = distances_to_centers[np.arange(len(endpoints)), nearest]
    nearest[nearest_distance > scheme.attach_eps] = -1
    line_clusters = nearest.reshape(-1, 2)
    line_snap = nearest_distance.reshape(-1, 2)

    candidates = {}
    for index, (first_cluster, second_cluster) in enumerate(line_clusters):
        if first_cluster < 0 or second_cluster < 0 or first_cluster == second_cluster:
            continue
        first, second = sorted((int(first_cluster), int(second_cluster)))
        key = (first, second)
        candidate = candidates.setdefault(
            key,
            {
                "votes": 0,
                "snap_sum": 0.0,
                "consistency_sum": 0.0,
                "proj_min": math.inf,
                "proj_max": -math.inf,
            },
        )
        graph_vector = centers[second] - centers[first]
        graph_length = np.linalg.norm(graph_vector)
        if graph_length <= 1e-12:
            continue
        graph_direction = graph_vector / graph_length
        line_direction = vectors[index] / max(lengths[index], 1e-12)
        consistency = abs(float(np.dot(line_direction, graph_direction)))
        if consistency < scheme.min_consistency:
            continue
        projections = (
            (lines[index] - centers[first].reshape(1, 3)) @ graph_direction
        ) / graph_length
        candidate["votes"] += 1
        candidate["snap_sum"] += float(line_snap[index].mean())
        candidate["consistency_sum"] += consistency
        candidate["proj_min"] = min(candidate["proj_min"], float(projections.min()))
        candidate["proj_max"] = max(candidate["proj_max"], float(projections.max()))

    if (
        scheme.path_pair_max_len > 0
        and scheme.path_support_radius > 0
        and edge_points.size != 0
    ):
        center_distances = cdist(centers, centers)
        for first in range(len(centers)):
            order = np.argsort(center_distances[first])
            order = (
                order[1 : scheme.path_knn + 1]
                if scheme.path_knn > 0
                else order[1:]
            )
            for second in order:
                if second <= first:
                    continue
                graph_length = float(center_distances[first, second])
                if graph_length < scheme.min_graph_len or graph_length > scheme.path_pair_max_len:
                    continue
                key = (first, int(second))
                if key in candidates and candidates[key]["votes"] >= scheme.min_votes:
                    continue
                support, coverage = point_segment_stats(
                    edge_points,
                    centers[first],
                    centers[second],
                    scheme.path_support_radius,
                )
                if support < scheme.path_min_support or coverage < scheme.path_min_coverage:
                    continue
                candidate = candidates.setdefault(
                    key,
                    {
                        "votes": 0,
                        "snap_sum": 0.0,
                        "consistency_sum": 0.0,
                        "proj_min": math.inf,
                        "proj_max": -math.inf,
                    },
                )
                candidate["path_support"] = max(
                    int(candidate.get("path_support", 0)), support
                )
                candidate["path_coverage"] = max(
                    float(candidate.get("path_coverage", 0.0)), coverage
                )

    scored = []
    for (first, second), candidate in candidates.items():
        votes = candidate["votes"]
        has_path = "path_support" in candidate
        if votes < scheme.min_votes and not has_path:
            continue
        graph_length = float(np.linalg.norm(centers[second] - centers[first]))
        if graph_length < scheme.min_graph_len:
            continue
        snap = candidate["snap_sum"] / max(votes, 1) if votes > 0 else 0.0
        if snap > scheme.max_snap:
            continue
        consistency = (
            candidate["consistency_sum"] / max(votes, 1)
            if votes > 0
            else 1.0
        )
        direct_coverage = max(
            0.0,
            min(1.0, candidate["proj_max"]) - max(0.0, candidate["proj_min"]),
        )
        support, edge_coverage = point_segment_stats(
            edge_points,
            centers[first],
            centers[second],
            scheme.edge_support_radius,
        )
        if has_path:
            support = max(support, int(candidate.get("path_support", 0)))
            edge_coverage = max(
                edge_coverage,
                float(candidate.get("path_coverage", 0.0)),
            )
        coverage = max(direct_coverage, edge_coverage)
        if support < scheme.min_edge_support:
            continue
        support_norm = min(
            1.0,
            support
            / max(4.0, graph_length / max(scheme.edge_support_radius, 1e-6)),
        )
        score = (
            scheme.score_vote_weight * math.log1p(votes)
            + consistency
            + 0.5 * coverage
            + scheme.score_support_weight * support_norm
            - scheme.score_snap_weight * (snap / max(scheme.attach_eps, 1e-6))
            + scheme.path_score_weight * float(has_path) * coverage
        )
        scored.append({"edge": (first, second), "score": score})

    scored.sort(key=lambda item: item["score"], reverse=True)
    chosen = []
    chosen_set = set()
    degrees = np.zeros(len(centers), dtype=np.int64)
    directions = [[] for _ in range(len(centers))]
    cosine_threshold = math.cos(math.radians(scheme.angle_sep_deg))

    for item in scored:
        first, second = item["edge"]
        if (first, second) in chosen_set:
            continue
        if degrees[first] >= scheme.max_degree or degrees[second] >= scheme.max_degree:
            continue
        vector = centers[second] - centers[first]
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            continue
        first_direction = vector / norm
        second_direction = -first_direction
        if any(
            float(np.dot(first_direction, old)) > cosine_threshold
            for old in directions[first]
        ):
            continue
        if any(
            float(np.dot(second_direction, old)) > cosine_threshold
            for old in directions[second]
        ):
            continue
        chosen.append((first, second))
        chosen_set.add((first, second))
        degrees[first] += 1
        degrees[second] += 1
        directions[first].append(first_direction)
        directions[second].append(second_direction)

    output = (
        np.asarray(
            [[centers[first], centers[second]] for first, second in chosen],
            dtype=np.float64,
        )
        if chosen
        else np.empty((0, 2, 3), dtype=np.float64)
    )
    if scheme.output_repeats > 1:
        output = np.repeat(output, scheme.output_repeats, axis=0)
    return output, {
        "num_clusters": int(len(centers)),
        "num_candidates": int(len(scored)),
        "num_chosen": int(len(chosen)),
    }
