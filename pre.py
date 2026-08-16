import argparse
import data
import os.path as op
import yaml
import network
import numpy as np
from dotted.collection import DottedDict

import torch.nn.functional as F
import sys
import os
import torch
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader


def linecloud_corner_support_nms(
    rays,
    corner_merge_dist=0.12,
    angle_nms_deg=10.0,
    max_rays_per_corner=3,
    strong_thr=0.75,
    top1_thr=0.85,
    top2_sum_thr=1.50,
    min_strong_rays=2
):
    if len(rays) == 0:
        return [], np.empty((0, 3)), np.empty((0, 3))

    pts = np.stack([r['start_pt'] for r in rays], axis=0)
    scores = np.array([r['score'] for r in rays], dtype=np.float64)
    order = np.argsort(-scores)

    clusters = []
    for idx in order:
        p, s = pts[idx], scores[idx]
        hit = -1

        for ci, c in enumerate(clusters):
            if np.linalg.norm(p - c['center']) < corner_merge_dist:
                hit = ci
                break

        if hit < 0:
            clusters.append({
                'center': p.copy(),
                'weight_sum': s,
                'ray_indices': [idx]
            })
        else:
            c = clusters[hit]
            w_old, w_new = c['weight_sum'], s
            c['center'] = (c['center'] * w_old + p * w_new) / (w_old + w_new + 1e-8)
            c['weight_sum'] += w_new
            c['ray_indices'].append(idx)

    final_lines = []
    final_src_corners = []
    final_all_corners = []

    for c in clusters:
        rs = [rays[i] for i in c['ray_indices']]
        rs.sort(key=lambda x: x['score'], reverse=True)

        ray_scores = np.array([r['score'] for r in rs], dtype=np.float64)
        top1 = float(ray_scores[0])
        top2_sum = float(ray_scores[:2].sum()) if len(ray_scores) >= 2 else top1
        num_strong = int(np.sum(ray_scores >= strong_thr))

        if not (top1 >= top1_thr or top2_sum >= top2_sum_thr or num_strong >= min_strong_rays):
            continue

        keep = []

        for r in rs:
            dup = False

            for kr in keep:
                angle = np.degrees(
                    np.arccos(
                        np.clip(np.dot(r['dir'], kr['dir']), -1.0, 1.0)
                    )
                )

                if angle < angle_nms_deg:
                    dup = True
                    break

            if not dup:
                keep.append(r)

            if len(keep) >= max_rays_per_corner:
                break

        if len(keep) == 0:
            continue

        final_src_corners.append(c['center'])

        for r in keep:
            final_lines.append((r['start_pt'], r['raw_end']))
            final_all_corners.extend([c['center'], r['raw_end']])

    if len(final_all_corners) > 0:
        final_all_corners = np.asarray(list({tuple(np.round(p, 4)) for p in final_all_corners}))
    else:
        final_all_corners = np.empty((0, 3))

    if len(final_src_corners) > 0:
        final_src_corners = np.asarray(final_src_corners)
    else:
        final_src_corners = np.empty((0, 3))

    return final_lines, final_src_corners, final_all_corners


def norm_points_to_real(points_norm, gt):

    device = points_norm.device

    points_norm = points_norm.to(dtype=torch.float64)

    n_center = gt['n_center'].to(device=device, dtype=torch.float64)
    n_factor = gt['n_factor'].to(device=device, dtype=torch.float64)

    while n_center.dim() < points_norm.dim():
        n_center = n_center.unsqueeze(0)

    while n_factor.dim() < points_norm.dim():
        n_factor = n_factor.unsqueeze(0)

    return points_norm / n_factor + n_center


def extract_gt_lines(gt):
    device = gt['corner_xyz'].device
    gt_lines_real = []

    if 'corner_link' not in gt or 'corner_xyz' not in gt:
        return gt_lines_real

    gt_corner_norm = gt['corner_xyz'].to(device).float()
    gt_corner_real = norm_points_to_real(gt_corner_norm, gt).detach().cpu().numpy()

    for link in gt['corner_link'].long():
        a = int(link[0].item())
        b = int(link[1].item())
        gt_lines_real.append((gt_corner_real[a], gt_corner_real[b]))

    return gt_lines_real


def extract_real_pc(model_input, gt):
    pc_norm = model_input['pc']
    pc_real = norm_points_to_real(pc_norm, gt)
    return pc_real.detach().cpu().numpy()


def extract_edge_pc(model_output, model_input, gt, edge_thr=0.5):
    pc_real = extract_real_pc(model_input, gt)

    edge_logits = model_output['edge_mask']
    edge_mask = (torch.sigmoid(edge_logits) > edge_thr).detach().cpu().numpy()

    return pc_real[edge_mask]


def make_lines_and_rays_from_full_AB(
    A_norm_full,
    B_norm_full,
    score_full,
    valid_mask,
    gt,
    score_thr=None,
    min_len=0.15
):

    device = A_norm_full.device

    if A_norm_full.numel() == 0 or B_norm_full.numel() == 0:
        return [], []

    if score_full is None:
        score_full = torch.ones(valid_mask.shape, dtype=torch.float32, device=device)

    if score_thr is None:
        final_mask = valid_mask
    else:
        final_mask = valid_mask & (score_full > score_thr)

    line_i, line_b = torch.where(final_mask)

    if line_i.shape[0] == 0:
        return [], []

    A_sel_norm = A_norm_full[line_i, line_b]
    B_sel_norm = B_norm_full[line_i, line_b]
    score_sel = score_full[line_i, line_b]

    A_real = norm_points_to_real(A_sel_norm, gt).detach().cpu().numpy()
    B_real = norm_points_to_real(B_sel_norm, gt).detach().cpu().numpy()
    scores = score_sel.detach().cpu().numpy()

    lines = []
    rays_for_nms = []

    for k in range(A_real.shape[0]):
        start_pt = A_real[k]
        end_pt = B_real[k]

        vec = end_pt - start_pt
        length = float(np.linalg.norm(vec))

        if length < min_len:
            continue

        direction = vec / (length + 1e-8)
        score = float(scores[k])

        lines.append((start_pt, end_pt))

        rays_for_nms.append({
            'start_pt': start_pt,
            'raw_end': end_pt,
            'dir': direction,
            'score': score
        })

    return lines, rays_for_nms


def extract_raw_topk_proposals(model_output, gt, min_len=0.15):

    corner_xyz = model_output['corner_xyz']
    ray_delta = model_output['ray_delta']
    ray_dist = model_output['ray_dist']
    ray_logits = model_output['ray_logits']
    candidate_mask = model_output['candidate_mask']

    device = corner_xyz.device
    S, Bins = ray_logits.shape

    if S == 0 or Bins == 0:
        return []

    flat_rays = gt['base_rays'].to(device)

    if flat_rays.dim() == 2:
        flat_rays_exp = flat_rays.unsqueeze(0).expand(S, -1, -1)
    else:
        flat_rays_exp = flat_rays

    direction = F.normalize(flat_rays_exp + ray_delta, dim=-1, eps=1e-6)

    A_full = corner_xyz.unsqueeze(1).expand(S, Bins, 3)
    B_full = A_full + ray_dist.unsqueeze(-1) * direction

    proposal_score = torch.sigmoid(ray_logits)

    lines, _ = make_lines_and_rays_from_full_AB(
        A_norm_full=A_full,
        B_norm_full=B_full,
        score_full=proposal_score,
        valid_mask=candidate_mask,
        gt=gt,
        score_thr=None,
        min_len=min_len
    )

    return lines


def find_history_index(model_output, stage_name):
    stages = model_output.get('history_stage', [])

    for idx, name in enumerate(stages):
        if name == stage_name:
            return idx

    return None


def extract_history_stage_lines(
    model_output,
    gt,
    stage_name,
    score_mode='step',
    stage_thr=0.75,
    min_len=0.15
):

    idx = find_history_index(model_output, stage_name)

    if idx is None:
        return [], []

    A_full = model_output['history_A'][idx]
    B_full = model_output['history_B'][idx]
    valid_mask = model_output['history_mask'][idx]

    if score_mode == 'round':
        score_full = model_output['history_round_scores'][idx]
    else:
        score_full = model_output['history_step_scores'][idx]

    return make_lines_and_rays_from_full_AB(
        A_norm_full=A_full,
        B_norm_full=B_full,
        score_full=score_full,
        valid_mask=valid_mask,
        gt=gt,
        score_thr=stage_thr,
        min_len=min_len
    )


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

def save_point_cloud_xyz(path, pc):

    if pc is None:
        pc = np.empty((0, 3), dtype=np.float64)

    pc = np.asarray(pc, dtype=np.float64)

    if pc.size == 0:
        open(path, 'w').close()
        return

    np.savetxt(path, pc, fmt='%.8f')

def save_lines_to_obj(path, lines):

    if lines is None or len(lines) == 0:
        open(path, 'w').close()
        return

    with open(path, 'w') as f:
        for start_pt, end_pt in lines:
            start_pt = np.asarray(start_pt, dtype=np.float64)
            end_pt = np.asarray(end_pt, dtype=np.float64)

            f.write(f"v {start_pt[0]:.8f} {start_pt[1]:.8f} {start_pt[2]:.8f}\n")
            f.write(f"v {end_pt[0]:.8f} {end_pt[1]:.8f} {end_pt[2]:.8f}\n")

        for i in range(len(lines)):
            f.write(f"l {2 * i + 1} {2 * i + 2}\n")


def load_end2end_model(log_path, device, checkpoint='final'):
    config_path = op.join(log_path, 'config.yaml')
    opt = yaml.safe_load(open(config_path))
    opt = DottedDict(opt)

    ckpt_name = 'model_final.pth' if checkpoint == 'final' else f'model_epoch_{int(checkpoint):04d}.pth'

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='torch.serialization')

    model = network.model.CornerExt(opt)
    ckpt = torch.load(op.join(log_path, f'checkpoints/{ckpt_name}'), map_location=device)

    if 'ema_state_dict' in ckpt:
        print("Loaded EMA model weights.")
        model.load_state_dict(ckpt['ema_state_dict'])
    else:
        print("EMA weights not found; using standard model weights.")
        model.load_state_dict(ckpt['model_state_dict'])

    model.to(device).eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Point2Contour inference on a prepared dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True, help="Prepared dataset directory")
    parser.add_argument("--model-dir", required=True, help="Training result directory")
    parser.add_argument("--checkpoint", default="final", help="Checkpoint name: final or an epoch number")
    parser.add_argument("--output-dir", default=None, help="Exact output directory; defaults to res_pre/<dataset>_<timestamp>")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val", help="Dataset split")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Inference device")
    parser.add_argument("--num-workers", type=int, default=1, help="Data loader workers")
    parser.add_argument("--max-points", type=int, default=1000000, help="Skip scenes above this point count; use 0 to disable")
    parser.add_argument("--refine-threshold", type=float, default=0.75, help="Line refinement probability threshold")
    parser.add_argument("--stage-threshold", type=float, default=0.75, help="Stage output probability threshold")
    parser.add_argument("--num-iters", type=int, default=3, help="Bidirectional refinement rounds")
    parser.add_argument("--min-line-length", type=float, default=0.15, help="Minimum output line length in meters")
    parser.add_argument("--corner-merge-distance", type=float, default=0.12, help="NMS corner merge distance in meters")
    parser.add_argument("--angle-nms-deg", type=float, default=10.0, help="NMS direction threshold in degrees")
    parser.add_argument("--max-rays-per-corner", type=int, default=3, help="Maximum retained rays per corner")
    parser.add_argument("--strong-ray-threshold", type=float, default=0.75, help="Strong ray score threshold")
    parser.add_argument("--top1-threshold", type=float, default=0.85, help="Single-ray corner acceptance threshold")
    parser.add_argument("--top2-sum-threshold", type=float, default=1.50, help="Two-ray corner acceptance threshold")
    parser.add_argument("--min-strong-rays", type=int, default=2, help="Strong rays required to retain a corner")
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


if __name__ == '__main__':
    args = parse_args()
    ROOT_path = os.path.dirname(os.path.abspath(__file__))
    device = resolve_device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(
        os.path.normpath(os.path.abspath(os.path.expanduser(args.data_root)))
    ) or "dataset"

    RESOUT_DIR = (
        os.path.abspath(os.path.expanduser(args.output_dir))
        if args.output_dir
        else os.path.join(ROOT_path, 'res_pre', f"{dataset_name}_{timestamp}")
    )
    os.makedirs(RESOUT_DIR, exist_ok=True)

    log_file_path = os.path.join(RESOUT_DIR, "inference_log.txt")
    sys.stdout = Logger(log_file_path)

    print("Starting inference.")
    print(f"Output directory: {RESOUT_DIR}")

    REFINE_THR = args.refine_threshold
    STAGE_THR = args.stage_threshold
    NUM_ITERS = args.num_iters
    MIN_LEN = args.min_line_length

    demo_config = {
        'mode': args.split
    }

    demo_dataset = data.dataset.NewDateset(demo_config, data_root=args.data_root)

    my_dataloader = DataLoader(
        demo_dataset,
        collate_fn=demo_dataset.collate_fn,
        batch_size=1,
        num_workers=args.num_workers
    )

    model = load_end2end_model(args.model_dir, device, checkpoint=args.checkpoint)

    with torch.no_grad():
        pbar = tqdm(my_dataloader, desc="Inference")

        for my_data in pbar:
            model_input, gt, info = my_data
            name = str(info['name'])

            pc_tensor = model_input['pc']

            num_points = pc_tensor.shape[1] if pc_tensor.dim() == 3 else pc_tensor.shape[0]

            if args.max_points > 0 and num_points > args.max_points:
                print(f"Skipping {name}: point count {num_points} exceeds the {args.max_points}-point limit.")
                continue  

            m_input_cuda = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in model_input.items()
            }

            gt_cuda = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in gt.items()
            }

            out = model.predict(
                m_input_cuda,
                refine_thresh=REFINE_THR,
                num_iters=NUM_ITERS
            )

            scene_dir = os.path.join(RESOUT_DIR, name)
            os.makedirs(scene_dir, exist_ok=True)

            real_pc = extract_real_pc(m_input_cuda, gt_cuda)
            lines_gt = extract_gt_lines(gt_cuda)
            edge_probability = torch.sigmoid(out['edge_mask']).detach().cpu().numpy()
            point_output = np.column_stack((real_pc, edge_probability))

            lines_raw_topk = extract_raw_topk_proposals(
                out,
                gt_cuda,
                min_len=0.0
            )

            lines_pred, final_rays = extract_history_stage_lines(
                out,
                gt_cuda,
                stage_name=f'iter_{NUM_ITERS - 1}_B_to_A',
                score_mode='round',
                stage_thr=STAGE_THR,
                min_len=MIN_LEN
            )

            lines_pred_nms, _, _ = linecloud_corner_support_nms(
                final_rays,
                corner_merge_dist=args.corner_merge_distance,
                angle_nms_deg=args.angle_nms_deg,
                max_rays_per_corner=args.max_rays_per_corner,
                strong_thr=args.strong_ray_threshold,
                top1_thr=args.top1_threshold,
                top2_sum_thr=args.top2_sum_threshold,
                min_strong_rays=args.min_strong_rays,
            )

            save_lines_to_obj(os.path.join(scene_dir, "gt_wire.obj"), lines_gt)
            save_point_cloud_xyz(os.path.join(scene_dir, "pc.xyz"), point_output)
            save_lines_to_obj(os.path.join(scene_dir, "raw_topk.obj"), lines_raw_topk)
            save_lines_to_obj(os.path.join(scene_dir, "pre_seg.obj"), lines_pred)
            save_lines_to_obj(os.path.join(scene_dir, "pre_seg_nms.obj"), lines_pred_nms)

            print(
                f"{name}: "
                f"gt={len(lines_gt)}, "
                f"raw_topk={len(lines_raw_topk)}, "
                f"pred={len(lines_pred)}, "
                f"pred_nms={len(lines_pred_nms)}"
            )

    print("Inference completed. Results were saved as OBJ and XYZ files.")
