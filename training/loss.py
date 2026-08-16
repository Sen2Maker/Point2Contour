import torch
import torch.nn as nn
import torch.nn.functional as F


class JointWireframeLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.alpha = 0.75
        self.gamma = 2.0

    def forward(self, model_output, gt, current_epoch=0, max_epochs=80):
        device = model_output['corner_xyz'].device

        logits_edge = model_output['edge_mask']
        loss_edge = F.binary_cross_entropy_with_logits(logits_edge, gt['edge_soft_label'].float())

        pos_idx = model_output['pos_idx'].long()
        S = pos_idx.shape[0]

        loss_corner_reg = torch.tensor(0.0, device=device)
        loss_corner2edge = torch.tensor(0.0, device=device)
        corner_xyz = model_output['corner_xyz']

        if S > 0:
            gt_xyz_S = gt['corner_xyz'].to(device).float()
            sampled_gt_idx = model_output.get('sampled_gt_idx', None)

            if sampled_gt_idx is not None:
                gt_xyz_matched = gt_xyz_S[sampled_gt_idx.long()]
                loss_corner_main = F.l1_loss(corner_xyz, gt_xyz_matched, reduction='mean')
                loss_corner_reg = loss_corner_main

                indptr = gt['corner_adj_indptr'].to(device).long()
                indices = gt['corner_adj_indices'].to(device).long()

                if indptr.shape[0] > 1:
                    start_idx = indptr[sampled_gt_idx.long()]
                    end_idx = indptr[sampled_gt_idx.long() + 1]
                    counts = end_idx - start_idx

                    MAX_NEIGHBORS = 32
                    offsets = torch.arange(MAX_NEIGHBORS, device=device).unsqueeze(0).expand(S, -1)
                    mask = offsets < counts.unsqueeze(1)
                    valid_flat_idx = (start_idx.unsqueeze(1) + offsets)[mask]

                    neighbors = indices[valid_flat_idx]
                    owner_s_idx = torch.repeat_interleave(torch.arange(S, device=device), counts)
                    owner_gt_idx = sampled_gt_idx[owner_s_idx]

                    A_pts = gt_xyz_S[owner_gt_idx]
                    B_pts = gt_xyz_S[neighbors]
                    P_pts = corner_xyz[owner_s_idx]

                    AB = B_pts - A_pts
                    AP = P_pts - A_pts
                    t = torch.sum(AP * AB, dim=-1) / (torch.sum(AB * AB, dim=-1) + 1e-12)
                    t = torch.clamp(t, 0.0, 1.0).unsqueeze(-1)
                    proj = A_pts + t * AB

                    dists = torch.norm(P_pts - proj, p=2, dim=-1)

                    num_valid_edges = mask.sum().float().clamp_min(1.0)
                    loss_corner2edge = dists.sum() / num_valid_edges

        ray_logits = model_output['ray_logits']
        ray_delta = model_output['ray_delta']
        ray_dist = model_output['ray_dist']
        flat_rays = gt['base_rays'].to(device)

        n_factor = gt['n_factor'].to(device).view(-1)[0].item() if 'n_factor' in gt else 1.0

        gt_cls = model_output.get('dyn_gt_cls', None)
        gt_reg = model_output.get('dyn_gt_reg', None)
        gt_neighbor_idx = model_output.get('dyn_gt_endpoint', None)

        if gt_cls is None:
            gt_cls = torch.zeros_like(ray_logits)
            gt_reg = torch.zeros_like(ray_delta[..., 0:4])
            gt_neighbor_idx = torch.full_like(ray_logits, -1, dtype=torch.long)

        pred_dir = F.normalize(flat_rays.unsqueeze(0) + ray_delta, dim=-1, eps=1e-12)

        pred_sigmoid = torch.sigmoid(ray_logits)
        bce_loss = F.binary_cross_entropy_with_logits(ray_logits, gt_cls, reduction='none')
        focal_weight = torch.abs(gt_cls - pred_sigmoid) ** self.gamma
        alpha_t = torch.where(gt_cls > 0.0, self.alpha, 1.0 - self.alpha)
        loss_ray_cls = (alpha_t * focal_weight * bce_loss).mean()

        smooth = 1e-5
        input_flat, target_flat = pred_sigmoid.view(-1), gt_cls.view(-1)
        intersection = (input_flat * target_flat).sum()
        loss_dice = 1.0 - (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)

        mask_pos = (gt_cls > 0.99)
        num_pos = mask_pos.sum().float().clamp_min(1.0)

        p_delta = ray_delta[mask_pos]
        t_delta = gt_reg[..., 0:3][mask_pos]

        loss_delta = F.smooth_l1_loss(p_delta, t_delta, reduction='sum') / num_pos

        p_dist = ray_dist[mask_pos]
        t_dist = gt_reg[..., 3][mask_pos]

        log_p_dist = torch.log(p_dist + 1.0)
        log_t_dist = torch.log(t_dist + 1.0)
        loss_dist_raw = F.smooth_l1_loss(log_p_dist, log_t_dist, reduction='sum') / num_pos

        pos_i, pos_b = torch.where(mask_pos)
        A = corner_xyz[pos_i]
        dir_vec = pred_dir[pos_i, pos_b]
        L = ray_dist[pos_i, pos_b].clamp(min=0.0).unsqueeze(-1)
        B = A + L * dir_vec

        gt_j = gt_neighbor_idx[pos_i, pos_b]
        valid_end = gt_j >= 0
        num_valid_end = valid_end.sum().float().clamp_min(1.0)

        global_corner_xyz = gt['corner_xyz'].to(device).float()
        gt_j_v = gt_j[valid_end]
        q_pred = B[valid_end]
        q_gt = global_corner_xyz[gt_j_v]

        loss_endpoint = F.l1_loss(q_pred, q_gt, reduction='sum') / num_valid_end

        train_refine_dict = model_output.get('train_refine_dict', None)

        loss_line_refine = torch.tensor(0.0, device=device)
        loss_inside = torch.tensor(0.0, device=device)
        loss_refined_dist = torch.tensor(0.0, device=device)
        loss_endpoint_ce = torch.tensor(0.0, device=device)
        loss_perp_offset = torch.tensor(0.0, device=device)

        if (train_refine_dict is not None) and ('line_targets' in train_refine_dict):
            inside_logits = train_refine_dict['support_logits']
            endpoint_logits = train_refine_dict['endpoint_logits']
            line_logits = train_refine_dict['line_logits']
            L_refined = train_refine_dict['L_refined']
            L_ext = train_refine_dict['L_ext'].detach()

            line_targets = train_refine_dict['line_targets']
            sampled_i = train_refine_dict['sampled_i']
            sampled_b = train_refine_dict['sampled_b']

            N_samples = inside_logits.shape[0]
            K_nodes = inside_logits.shape[1]

            if N_samples > 0:
                loss_line_refine = F.binary_cross_entropy_with_logits(line_logits.squeeze(-1), line_targets)

                pos_mask = (line_targets > 0.5)
                pos_i = sampled_i[pos_mask]
                pos_b = sampled_b[pos_mask]

                gt_neighbor_idx = model_output.get('dyn_gt_endpoint', None)
                if gt_neighbor_idx is not None:
                    gt_j = gt_neighbor_idx[pos_i, pos_b]
                    valid_gt = (gt_j >= 0)

                    gt_corner_xyz_all = gt['corner_xyz'].to(device).float()
                    q_gt = gt_corner_xyz_all[gt_j[valid_gt]]

                    A_pred = corner_xyz[sampled_i].detach()
                    dir_pred = pred_dir[sampled_i, sampled_b].detach()

                    A_pos = A_pred[pos_mask][valid_gt]
                    dir_pos = dir_pred[pos_mask][valid_gt]
                    L_ext_pos = L_ext[pos_mask][valid_gt]
                    L_refined_pos = L_refined[pos_mask][valid_gt]

                    v = q_gt - A_pos
                    t_gt = (v * dir_pos).sum(dim=-1)

                    closest = A_pos + t_gt.unsqueeze(-1) * dir_pos
                    perp_dist = torch.norm(closest - q_gt, dim=-1)

                    dist_mask = (t_gt > 0) & (t_gt < L_ext_pos) & (perp_dist < 0.25 * n_factor)
                    offset_mask = (t_gt > 0) & (t_gt < L_ext_pos) & (perp_dist < 0.60 * n_factor)

                    num_dist_mask = dist_mask.sum().float().clamp_min(1.0)
                    num_offset_mask = offset_mask.sum().float().clamp_min(1.0)

                    valid_t_gt = t_gt[dist_mask].unsqueeze(1)
                    valid_L_ext = L_ext_pos[dist_mask]
                    alphas = torch.linspace(0.0, 1.0, steps=K_nodes, device=device)
                    valid_node_dists = alphas.view(1, -1) * valid_L_ext.view(-1, 1)

                    loss_refined_dist = F.smooth_l1_loss(L_refined_pos[dist_mask], t_gt[dist_mask],
                                                         reduction='sum') / num_dist_mask

                    valid_inside_logits = inside_logits[pos_mask][valid_gt][dist_mask]
                    transition = 0.10 * n_factor
                    inside_targets = torch.sigmoid((valid_t_gt - valid_node_dists) / (transition + 1e-6))

                    loss_inside = F.binary_cross_entropy_with_logits(valid_inside_logits, inside_targets,
                                                                     reduction='sum') / (num_dist_mask * K_nodes)

                    valid_endpoint_logits = endpoint_logits[pos_mask][valid_gt][dist_mask]
                    node_step = valid_L_ext / (K_nodes - 1)
                    physical_sigma_norm = 0.10 * n_factor
                    target_sigma = torch.maximum(0.5 * node_step,
                                                 torch.tensor(physical_sigma_norm, device=device)).unsqueeze(1)
                    endpoint_targets = torch.exp(-(valid_node_dists - valid_t_gt) ** 2 / (2 * target_sigma ** 2))
                    endpoint_targets = endpoint_targets / (endpoint_targets.sum(dim=1, keepdim=True) + 1e-8)
                    log_probs = F.log_softmax(valid_endpoint_logits, dim=-1)

                    loss_endpoint_ce = -(endpoint_targets * log_probs).sum(dim=-1).sum() / num_dist_mask

                    q_gt_offset = q_gt[offset_mask]
                    closest_offset = closest[offset_mask]
                    perp_target = q_gt_offset - closest_offset
                    delta_B_perp_pred = train_refine_dict['delta_B_perp'][pos_mask][valid_gt][offset_mask]

                    loss_perp_offset = F.l1_loss(delta_B_perp_pred, perp_target, reduction='sum') / num_offset_mask

        computed_losses = {
            'edge': loss_edge,
            'corner_xyz': loss_corner_reg,
            'corner2edge': loss_corner2edge,

            'loss_cls': loss_ray_cls,
            'loss_dice': loss_dice,

            'loss_dir': loss_delta,
            'loss_dist': loss_dist_raw,
            'loss_endpoint': loss_endpoint,

            'loss_line_refine': loss_line_refine,
            'loss_refined_dist': loss_refined_dist,
            'loss_endpoint_ce': loss_endpoint_ce,
            'loss_inside': loss_inside,
            'loss_perp_offset': loss_perp_offset
        }

        total_loss = torch.tensor(0.0, device=device)
        raw_loss_dict = {}

        for loss_name, raw_value in computed_losses.items():

            base_factor = self.opt.loss[loss_name].factor
            raw_loss_dict[loss_name] = raw_value.detach()
            total_loss += raw_value * base_factor

        return total_loss, raw_loss_dict
