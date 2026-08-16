from branch import *


from CrossAttention import *

import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import math


class CornerExt(nn.Module):

    def __init__(self, opt):
        super(CornerExt, self).__init__()
        self.encoder = PointSimpleEncoder(opt.model.encoder)
        self.edge_decoder = EdgeDecoder(opt.model.edge_decoder)

        self.joint_decoder = JointCornerRayDecoderSimple(opt.model.joint_decoder)

        self.line_refiner = LineRefineNetwork(opt.model.line_refiner)

    @torch.no_grad()  
    def _compute_dynamic_ray_targets(self, pred_xyz, sampled_gt_idx, gt_corner_xyz, indptr, indices, base_rays):
        device = pred_xyz.device
        S = pred_xyz.shape[0]  
        B = base_rays.shape[0]

        dyn_gt_cls = torch.zeros((S, B), dtype=torch.float32, device=device)
        dyn_gt_reg = torch.zeros((S, B, 4), dtype=torch.float32, device=device)
        dyn_gt_endpoint_idx = torch.full((S, B), -1, dtype=torch.long, device=device)

        if S == 0 or indptr.shape[0] <= 1:
            return dyn_gt_cls, dyn_gt_reg, dyn_gt_endpoint_idx

        start_idx = indptr[sampled_gt_idx]
        end_idx = indptr[sampled_gt_idx + 1]
        counts = end_idx - start_idx  

        MAX_NEIGHBORS = 32
        counts_safe = counts.clamp(max=MAX_NEIGHBORS)

        offsets = torch.arange(MAX_NEIGHBORS, device=device).unsqueeze(0).expand(S, -1)

        mask = offsets < counts_safe.unsqueeze(1)
        valid_flat_idx = (start_idx.unsqueeze(1) + offsets)[mask]

        neighbors = indices[valid_flat_idx]

        owner = torch.repeat_interleave(torch.arange(S, device=device), counts_safe)

        vec = gt_corner_xyz[neighbors] - pred_xyz[owner]
        length = torch.norm(vec, dim=-1)

        valid = length > 1e-6

        owner, neighbors, vec, length = owner[valid], neighbors[valid], vec[valid], length[valid]

        dir_norm = vec / length.unsqueeze(-1)
        cos_sim = torch.matmul(dir_norm, base_rays.T)

        SIGMA_DEG = 20.0
        SOFT_THRESH_DEG = 45.0
        SOFT_THRESH_COS = math.cos(math.radians(SOFT_THRESH_DEG))
        gaussian_scale = -1.0 / (2.0 * math.radians(SIGMA_DEG) ** 2)

        valid_cos = torch.clamp(cos_sim, min=-0.9999, max=0.9999)
        angles_rad = torch.acos(valid_cos)
        scores = torch.exp(gaussian_scale * (angles_rad ** 2))

        mask_cos = cos_sim > SOFT_THRESH_COS
        scores = scores * mask_cos.float()

        dyn_gt_cls.scatter_reduce_(0, owner.unsqueeze(1).expand(-1, B), scores, reduce='amax', include_self=False)

        best_cos, best_b = torch.max(cos_sim, dim=-1)

        sort_idx = torch.argsort(best_cos)

        owner_s = owner[sort_idx]
        best_b_s = best_b[sort_idx]
        neighbors_s = neighbors[sort_idx]
        dir_norm_s = dir_norm[sort_idx]
        length_s = length[sort_idx]

        dyn_gt_cls[owner_s, best_b_s] = 1.0
        dyn_gt_reg[owner_s, best_b_s, 0:3] = dir_norm_s - base_rays[best_b_s]
        dyn_gt_reg[owner_s, best_b_s, 3] = length_s
        dyn_gt_endpoint_idx[owner_s, best_b_s] = neighbors_s

        return dyn_gt_cls, dyn_gt_reg, dyn_gt_endpoint_idx

    def _process_refiner(self, mode, ray_logits, ray_delta, ray_dist, flat_rays, corner_offset_pos, edge_xyz,
                         edge_feat, dyn_gt_cls=None, dyn_gt_endpoint=None, gt_corner_xyz=None):
        S, Bins = ray_logits.shape
        device = ray_logits.device

        train_refine_dict = None
        eval_refine_dict = {
            'candidate_mask': torch.zeros((S, Bins), dtype=torch.bool, device=device) if S > 0 else torch.zeros(
                (0, Bins), dtype=torch.bool, device=device),
            'line_logits': torch.full((S, Bins), -1e4, device=device) if S > 0 else torch.zeros((0, Bins),
                                                                                                device=device),
            'L_refined': ray_dist.clone(),
            'delta_B_perp': torch.zeros((S, Bins, 3), device=device) if S > 0 else torch.zeros((0, Bins, 3),
                                                                                               device=device)
        }

        if S == 0:
            return train_refine_dict, eval_refine_dict

        ray_probs = torch.sigmoid(ray_logits)

        if mode == 'train' and dyn_gt_cls is not None:
            pos_i, pos_b = torch.where(dyn_gt_cls > 0.99)
            neg_i, neg_b = torch.where((dyn_gt_cls < 0.5) & (ray_probs > 0.05))

            num_pos, num_neg = pos_i.shape[0], neg_i.shape[0]

            if num_neg > num_pos and num_pos > 0:
                prob = float(num_pos) / num_neg
                rand_mask = torch.rand(num_neg, device=device) < prob
                neg_i, neg_b = neg_i[rand_mask], neg_b[rand_mask]
            elif num_pos == 0 and num_neg > 0:
                prob = min(32.0 / num_neg, 1.0)
                rand_mask = torch.rand(num_neg, device=device) < prob
                neg_i, neg_b = neg_i[rand_mask], neg_b[rand_mask]

            final_i = torch.cat([pos_i, neg_i])
            final_b = torch.cat([pos_b, neg_b])
            final_targets = torch.cat(
                [torch.ones(pos_i.shape[0], device=device), torch.zeros(neg_i.shape[0], device=device)])

            if final_i.shape[0] > 0:
                A_sampled = corner_offset_pos[final_i]
                dir_sampled = F.normalize(flat_rays[final_b] + ray_delta[final_i, final_b], dim=-1)
                dist_sampled = ray_dist[final_i, final_b]

                L_ext_override = torch.clamp(1.5 * dist_sampled, max=2.0).detach()

                if dyn_gt_endpoint is not None:
                    pos_mask = (final_targets > 0.5)

                    p_i = final_i[pos_mask]
                    p_b = final_b[pos_mask]
                    gt_j = dyn_gt_endpoint[p_i, p_b]

                    valid_j = (gt_j >= 0)

                    q_gt = gt_corner_xyz[gt_j[valid_j]]
                    A_pos = A_sampled[pos_mask][valid_j]
                    dir_pos = dir_sampled[pos_mask][valid_j]

                    t_gt = ((q_gt - A_pos) * dir_pos).sum(dim=-1)

                    safe_L_ext = torch.maximum(1.5 * dist_sampled[pos_mask][valid_j].detach(), 1.1 * t_gt.detach())

                    temp_override = L_ext_override[pos_mask]

                    temp_override[valid_j] = safe_L_ext
                    L_ext_override[pos_mask] = temp_override

                refine_out = self.line_refiner(
                    A=A_sampled.detach(),
                    pred_dir=dir_sampled.detach(),
                    L_coarse=dist_sampled.detach(),
                    edge_xyz=edge_xyz.detach(),
                    edge_feat=edge_feat.detach(),
                    L_ext_override=L_ext_override
                )

                train_refine_dict = refine_out
                train_refine_dict['line_targets'] = final_targets
                train_refine_dict['sampled_i'] = final_i
                train_refine_dict['sampled_b'] = final_b

        else:
            K_proposals = min(6, Bins)
            _, topk_indices = torch.topk(ray_probs, K_proposals, dim=-1)
            eval_refine_dict['candidate_mask'].scatter_(1, topk_indices, True)

            pos_i, pos_b = torch.where(eval_refine_dict['candidate_mask'])

            if pos_i.shape[0] > 0:
                A_candidates = corner_offset_pos[pos_i]
                dir_candidates = F.normalize(flat_rays[pos_b] + ray_delta[pos_i, pos_b], dim=-1)
                dist_candidates = ray_dist[pos_i, pos_b]

                refine_out = self.line_refiner(
                    A=A_candidates.detach(),
                    pred_dir=dir_candidates.detach(),
                    L_coarse=dist_candidates.detach(),
                    edge_xyz=edge_xyz.detach(),
                    edge_feat=edge_feat.detach()
                )

                eval_refine_dict['line_logits'][pos_i, pos_b] = refine_out['line_logits'].squeeze(-1)
                eval_refine_dict['L_refined'][pos_i, pos_b] = refine_out['L_refined']
                eval_refine_dict['delta_B_perp'][pos_i, pos_b] = refine_out['delta_B_perp']

        return train_refine_dict, eval_refine_dict

    def forward(self, model_input, mode='train'):

        token_context, m2, m3 = self.encoder(model_input)

        feat_teach, edge_logits = self.edge_decoder(
            point_feat=m3,
            token_context=token_context,
            fp_idx=model_input['fp_idx'],
            fp_dist=model_input['fp_dist']
        )
        edge_logits = edge_logits.squeeze(-1)

        point_xyz = model_input['pc']
        if mode == 'train':
            edge_gt_mask = (model_input['edge_soft_label'] > 0.5)
            edge_xyz = point_xyz[edge_gt_mask]
            edge_feat = feat_teach[edge_gt_mask]
        else:
            edge_pred_mask = (torch.sigmoid(edge_logits) > 0.5)
            edge_xyz = point_xyz[edge_pred_mask]
            edge_feat = feat_teach[edge_pred_mask]

        gt_corner_xyz = model_input['corner_xyz'] if mode == 'train' else None
        centers_idx = model_input['centers_idx'].long()
        center_xyz = point_xyz[centers_idx]
        center_feat = m3[centers_idx]

        joint_out = self.joint_decoder(
            center_xyz=center_xyz,
            center_feat=center_feat,
            token_context=token_context,
            edge_xyz=edge_xyz,
            edge_feat=edge_feat,
            flat_rays=model_input['flat_rays'],
            gt_corner_xyz=gt_corner_xyz,
            mode=mode
        )

        corner_offset_pos = joint_out['corner_offset_pos']
        sampled_gt_idx = joint_out.get('sampled_gt_idx', None)

        dyn_gt_cls, dyn_gt_reg, dyn_gt_endpoint = None, None, None

        if mode == 'train':
            global_corner_xyz = model_input['corner_xyz']
            indptr = model_input['corner_adj_indptr']
            indices = model_input['corner_adj_indices']
            flat_rays = model_input['flat_rays']

            dyn_gt_cls, dyn_gt_reg, dyn_gt_endpoint = self._compute_dynamic_ray_targets(
                pred_xyz=corner_offset_pos.detach(),
                sampled_gt_idx=sampled_gt_idx,
                gt_corner_xyz=global_corner_xyz,
                indptr=indptr,
                indices=indices,
                base_rays=flat_rays
            )

        train_refine_dict, eval_refine_dict = self._process_refiner(
            mode=mode,
            ray_logits=joint_out['ray_logits'],
            ray_delta=joint_out['ray_delta'],
            ray_dist=joint_out['ray_dist'],
            flat_rays=model_input['flat_rays'],
            corner_offset_pos=corner_offset_pos,
            edge_xyz=edge_xyz,
            edge_feat=edge_feat,
            dyn_gt_cls=dyn_gt_cls,
            dyn_gt_endpoint=dyn_gt_endpoint,
            gt_corner_xyz=gt_corner_xyz
        )

        return {
            'edge_mask': edge_logits,

            'corner_xyz': corner_offset_pos,
            'corner_xyz_all': joint_out['corner_xyz_all'],
            'pred2gt_min_dist': joint_out.get('pred2gt_min_dist', None),
            'sampled_gt_idx': sampled_gt_idx,
            'pos_idx': joint_out['active_idx'],
            'feat': feat_teach,

            'ray_logits': joint_out['ray_logits'],
            'ray_delta': joint_out['ray_delta'],
            'ray_dist': joint_out['ray_dist'],

            'dyn_gt_cls': dyn_gt_cls,
            'dyn_gt_reg': dyn_gt_reg,
            'dyn_gt_endpoint': dyn_gt_endpoint,

            'ray_refine_logits': eval_refine_dict['line_logits'],
            'ray_dist_refined': eval_refine_dict['L_refined'],

            'candidate_mask': eval_refine_dict['candidate_mask'],
            'train_refine_dict': train_refine_dict,
            'eval_refine_dict': eval_refine_dict,
        }

    @torch.no_grad()
    def predict(self, model_input, refine_thresh=0.75, num_iters=3):

        token_context, m2, m3 = self.encoder(model_input)

        feat_teach, edge_logits = self.edge_decoder(
            point_feat=m3,
            token_context=token_context,
            fp_idx=model_input['fp_idx'],
            fp_dist=model_input['fp_dist']
        )
        edge_logits = edge_logits.squeeze(-1)

        point_xyz = model_input['pc']

        edge_probs = torch.sigmoid(edge_logits)
        edge_pred_mask = edge_probs > 0.5

        if edge_pred_mask.sum() == 0:
            k = max(1, point_xyz.shape[0] // 2)
            edge_idx = torch.topk(edge_probs, k=k, dim=0).indices

            edge_xyz = point_xyz[edge_idx]
            edge_feat = feat_teach[edge_idx]

            print(f"No edge points predicted; using the top {k} points.", flush=True)
        else:
            edge_xyz = point_xyz[edge_pred_mask]
            edge_feat = feat_teach[edge_pred_mask]

        centers_idx = model_input['centers_idx'].long()
        center_xyz = point_xyz[centers_idx]
        center_feat = m3[centers_idx]
        flat_rays = model_input['flat_rays']

        joint_out = self.joint_decoder(
            center_xyz=center_xyz,
            center_feat=center_feat,
            token_context=token_context,
            edge_xyz=edge_xyz,
            edge_feat=edge_feat,
            flat_rays=flat_rays,
            gt_corner_xyz=None,
            mode='val'
        )

        corner_xyz = joint_out['corner_offset_pos']
        ray_logits = joint_out['ray_logits']
        ray_delta = joint_out['ray_delta']
        ray_dist = joint_out['ray_dist']

        S, Bins = ray_logits.shape
        device = corner_xyz.device

        candidate_mask = torch.zeros((S, Bins), dtype=torch.bool, device=device)

        line_logits_latest = torch.full((S, Bins), -1e4, device=device)
        line_logits_fwd = torch.full((S, Bins), -1e4, device=device)
        line_logits_back = torch.full((S, Bins), -1e4, device=device)

        delta_B_perp_full = torch.zeros((S, Bins, 3), device=device)
        delta_A_perp_full = torch.zeros((S, Bins, 3), device=device)

        out = {
            'edge_mask': edge_logits,
            'feat': feat_teach,

            'corner_xyz': corner_xyz,
            'corner_xyz_all': joint_out.get('corner_xyz_all', None),
            'pos_idx': joint_out['active_idx'],
            'corner_logits': joint_out.get('corner_logits', None),

            'ray_logits': ray_logits,
            'ray_delta': ray_delta,
            'ray_dist': ray_dist,

            'candidate_mask': candidate_mask,
            'ray_refine_logits': line_logits_latest,
            'ray_dist_refined': ray_dist.clone(),

            'eval_refine_dict': {
                'candidate_mask': candidate_mask,
                'line_logits': line_logits_latest,
                'line_logits_fwd': line_logits_fwd,
                'line_logits_back': line_logits_back,
                'L_refined': ray_dist.clone(),
                'delta_B_perp': delta_B_perp_full,
                'delta_A_perp': delta_A_perp_full,
            },


            'history_A': [],
            'history_B': [],
            'history_logits': [],


            'history_step_logits': [],
            'history_step_scores': [],


            'history_round_scores': [],

            'history_stage': [],
            'history_mask': [],
        }

        if S == 0 or Bins == 0:
            A_empty = corner_xyz.unsqueeze(1).expand(S, Bins, 3).clone()
            B_empty = A_empty.clone()

            out['eval_refine_dict']['A_refined'] = A_empty
            out['eval_refine_dict']['B_refined'] = B_empty
            out['eval_refine_dict']['line_logits'] = line_logits_latest
            out['ray_refine_logits'] = line_logits_latest
            return out

        ray_probs = torch.sigmoid(ray_logits)
        K_proposals = min(6, Bins)

        if K_proposals > 0:
            _, topk_indices = torch.topk(ray_probs, K_proposals, dim=-1)
            candidate_mask.scatter_(1, topk_indices, True)

        if flat_rays.dim() == 2:
            flat_rays_exp = flat_rays.unsqueeze(0).expand(S, -1, -1)
        else:
            flat_rays_exp = flat_rays

        dir_init = F.normalize(flat_rays_exp + ray_delta, dim=-1, eps=1e-6)

        A_full = corner_xyz.unsqueeze(1).expand(S, Bins, 3).clone()
        B_full = A_full + ray_dist.unsqueeze(-1) * dir_init

        active_mask = candidate_mask.clone()

        def record_history(stage_name, step_mask, step_logits_full, round_score_full=None):
            step_scores_full = torch.sigmoid(step_logits_full)

            out['history_A'].append(A_full.clone())
            out['history_B'].append(B_full.clone())
            out['history_logits'].append(line_logits_latest.clone())

            out['history_step_logits'].append(step_logits_full.clone())
            out['history_step_scores'].append(step_scores_full.clone())

            if round_score_full is None:
                out['history_round_scores'].append(torch.zeros_like(step_scores_full))
            else:
                out['history_round_scores'].append(round_score_full.clone())

            out['history_stage'].append(stage_name)
            out['history_mask'].append(step_mask.clone())

        num_iters = max(int(num_iters), 0)

        for it in range(num_iters):

            fwd_i, fwd_b = torch.where(active_mask)

            if fwd_i.shape[0] == 0:
                break

            A_src = A_full[fwd_i, fwd_b]
            B_old = B_full[fwd_i, fwd_b]

            vec_fwd = B_old - A_src
            L_coarse_fwd = torch.norm(vec_fwd, dim=-1).clamp_min(1e-6)
            dir_fwd = vec_fwd / L_coarse_fwd.unsqueeze(-1)

            fwd_out = self.line_refiner(
                A=A_src.detach(),
                pred_dir=dir_fwd.detach(),
                L_coarse=L_coarse_fwd.detach(),
                edge_xyz=edge_xyz.detach(),
                edge_feat=edge_feat.detach()
            )

            L_fwd = fwd_out['L_refined']
            dB_fwd = fwd_out['delta_B_perp']
            logit_fwd = fwd_out['line_logits'].squeeze(-1)

            B_new = A_src + L_fwd.unsqueeze(-1) * dir_fwd + dB_fwd

            B_full[fwd_i, fwd_b] = B_new
            line_logits_fwd[fwd_i, fwd_b] = logit_fwd
            line_logits_latest[fwd_i, fwd_b] = logit_fwd
            delta_B_perp_full[fwd_i, fwd_b] = dB_fwd

            fwd_step_logits = torch.full((S, Bins), -1e4, device=device)
            fwd_step_logits[fwd_i, fwd_b] = logit_fwd

            fwd_step_mask = torch.zeros_like(candidate_mask)
            fwd_step_mask[fwd_i, fwd_b] = True

            record_history(
                stage_name=f'iter_{it}_A_to_B',
                step_mask=fwd_step_mask,
                step_logits_full=fwd_step_logits,
                round_score_full=None
            )

            score_fwd = torch.sigmoid(logit_fwd)
            pass_fwd = score_fwd > refine_thresh

            back_mask = torch.zeros_like(candidate_mask)
            if pass_fwd.shape[0] > 0:
                back_mask[fwd_i[pass_fwd], fwd_b[pass_fwd]] = True

            back_i, back_b = torch.where(back_mask)

            if back_i.shape[0] == 0:
                active_mask = back_mask
                continue

            A_old = A_full[back_i, back_b]
            B_src = B_full[back_i, back_b]

            vec_back = A_old - B_src
            L_coarse_back = torch.norm(vec_back, dim=-1).clamp_min(1e-6)
            dir_back = vec_back / L_coarse_back.unsqueeze(-1)

            back_out = self.line_refiner(
                A=B_src.detach(),
                pred_dir=dir_back.detach(),
                L_coarse=L_coarse_back.detach(),
                edge_xyz=edge_xyz.detach(),
                edge_feat=edge_feat.detach()
            )

            L_back = back_out['L_refined']
            dA_back = back_out['delta_B_perp']
            logit_back = back_out['line_logits'].squeeze(-1)

            A_new = B_src + L_back.unsqueeze(-1) * dir_back + dA_back

            A_full[back_i, back_b] = A_new
            line_logits_back[back_i, back_b] = logit_back
            line_logits_latest[back_i, back_b] = logit_back
            delta_A_perp_full[back_i, back_b] = dA_back

            back_step_logits = torch.full((S, Bins), -1e4, device=device)
            back_step_logits[back_i, back_b] = logit_back

            back_step_mask = torch.zeros_like(candidate_mask)
            back_step_mask[back_i, back_b] = True

            score_back = torch.sigmoid(logit_back)

            fwd_score_full = torch.sigmoid(line_logits_fwd)
            back_score_full = torch.sigmoid(back_step_logits)

            round_score_full = torch.sqrt(fwd_score_full * back_score_full + 1e-8)

            record_history(
                stage_name=f'iter_{it}_B_to_A',
                step_mask=back_step_mask,
                step_logits_full=back_step_logits,
                round_score_full=round_score_full
            )

            round_score_valid = torch.sqrt(score_fwd[pass_fwd] * score_back + 1e-8)
            pass_round = round_score_valid > refine_thresh

            next_active_mask = torch.zeros_like(candidate_mask)
            if pass_round.shape[0] > 0:
                next_active_mask[back_i[pass_round], back_b[pass_round]] = True

            active_mask = next_active_mask

        if len(out['history_A']) == 0:
            init_logits = torch.full((S, Bins), -1e4, device=device)
            init_mask = torch.zeros_like(candidate_mask)

            record_history(
                stage_name='init_no_refine',
                step_mask=init_mask,
                step_logits_full=init_logits,
                round_score_full=None
            )

        final_L = torch.norm(B_full - A_full, dim=-1)

        out['eval_refine_dict']['A_refined'] = A_full
        out['eval_refine_dict']['B_refined'] = B_full
        out['eval_refine_dict']['line_logits'] = line_logits_latest
        out['eval_refine_dict']['line_logits_fwd'] = line_logits_fwd
        out['eval_refine_dict']['line_logits_back'] = line_logits_back
        out['eval_refine_dict']['L_refined'] = final_L
        out['eval_refine_dict']['delta_B_perp'] = delta_B_perp_full
        out['eval_refine_dict']['delta_A_perp'] = delta_A_perp_full

        out['ray_refine_logits'] = line_logits_latest
        out['ray_dist_refined'] = final_L
        out['candidate_mask'] = candidate_mask

        return out

    @torch.no_grad()
    def forward_moni(self, model_output, gt, ray_threshold=0.5):

        metrics = {}
        device = model_output['edge_mask'].device

        n_center = gt['n_center'].to(device)
        n_factor = gt['n_factor'].to(device)

        if n_factor.ndim == 1: n_factor = n_factor.view(-1, 1)
        if n_center.ndim == 1: n_center = n_center.view(1, 3)

        pred_e = (torch.sigmoid(model_output['edge_mask']) > 0.5)
        gt_e_soft = (gt['edge_soft_label'].reshape(-1) > 0.5)

        rec_s, prec_s, f1_s, iou_s, acc_s = self.val_cube_pred(pred_e, gt_e_soft)

        metrics.update({
            'Edge/Soft_Recall': rec_s, 'Edge/Soft_Precision': prec_s, 'Edge/Soft_F1': f1_s,
        })

        pred_xyz_all_norm = model_output['corner_xyz_all']  
        pred_xyz_all_real = pred_xyz_all_norm / n_factor + n_center

        gt_xyz_global_norm = gt['corner_xyz'].to(device)
        gt_xyz_global_real = gt_xyz_global_norm / n_factor + n_center  

        pos_idx = model_output['pos_idx'].long()  
        N_all = pred_xyz_all_real.shape[0]

        metrics.update({
            'Corner_Cls/Precision': 0.0,
            'Corner_Cls/Recall': 0.0,
            'Corner_Cls/F1': 0.0,
        })

        corner_tp_mask = torch.zeros(pos_idx.shape[0], dtype=torch.bool, device=device)
        metrics.update({
            'Corner/Reg_Recall_0.5m': 0.0,
            'Corner/Avg_Loc_Error_Global(m)': 0.0,
            'Corner/Outlier_Ratio(>0.5m)': 0.0,
            'Corner/Avg_Loc_Error_TP(m)': 0.0
        })

        if pred_xyz_all_real.shape[0] > 0 and gt_xyz_global_real.shape[0] > 0:
            dists_full = torch.cdist(pred_xyz_all_real, gt_xyz_global_real, p=2.0)
            pred_ind_np, gt_ind_np = linear_sum_assignment(dists_full.detach().cpu().numpy())
            pred_ind = torch.as_tensor(pred_ind_np, device=device).long()
            gt_ind = torch.as_tensor(gt_ind_np, device=device).long()
            matched_dists = dists_full[pred_ind, gt_ind]
            is_success = matched_dists < 0.5
            tp_corners = is_success.sum().float()

            corner_precision = tp_corners / max(pred_xyz_all_real.shape[0], 1)
            corner_recall = tp_corners / max(gt_xyz_global_real.shape[0], 1)
            corner_f1 = 2 * corner_precision * corner_recall / (
                corner_precision + corner_recall + 1e-8
            )
            global_err = matched_dists.mean().item() if matched_dists.numel() else 0.0
            outlier_ratio = (~is_success).float().mean().item() if matched_dists.numel() else 0.0
            pure_loc_err_tp = matched_dists[is_success].mean().item() if is_success.any() else 0.0

            full_tp_mask = torch.zeros(N_all, dtype=torch.bool, device=device)
            full_tp_mask[pred_ind[is_success]] = True
            corner_tp_mask = full_tp_mask[pos_idx]

            metrics.update({
                'Corner_Cls/Precision': corner_precision.item(),
                'Corner_Cls/Recall': corner_recall.item(),
                'Corner_Cls/F1': corner_f1.item(),
                'Corner/Reg_Recall_0.5m': corner_recall.item(),
                'Corner/Avg_Loc_Error_Global(m)': global_err,
                'Corner/Outlier_Ratio(>0.5m)': outlier_ratio,
                'Corner/Avg_Loc_Error_TP(m)': pure_loc_err_tp
            })

        metrics.update({
            'Corner/Recall_Strict_<0.3m': 0.0,
            'Corner/Recall_Loose_<0.6m': 0.0,
        })

        if pred_xyz_all_real.shape[0] > 0 and gt_xyz_global_real.shape[0] > 0:
            dists = torch.cdist(pred_xyz_all_real, gt_xyz_global_real, p=2.0)
            min_dist_gt_to_pred, _ = torch.min(dists, dim=0)

            recall_strict = (min_dist_gt_to_pred < 0.3).float().mean().item()
            recall_loose = (min_dist_gt_to_pred < 0.6).float().mean().item()

            metrics.update({
                'Corner/Recall_Strict_<0.3m': recall_strict,
                'Corner/Recall_Loose_<0.6m': recall_loose,
            })

        metrics.update({
            'Ray_Stage1/Precision': 0.0, 'Ray_Stage1/Recall': 0.0, 'Ray_Stage1/F1': 0.0, 'Ray_Stage1/TP_Count': 0.0,
            'Ray_Refine_Train/Precision': 0.0, 'Ray_Refine_Train/Recall': 0.0, 'Ray_Refine_Train/F1': 0.0,
            'Ray_Final_Cascade/Precision': 0.0, 'Ray_Final_Cascade/Recall': 0.0, 'Ray_Final_Cascade/F1': 0.0,
            'Ray_Geom/Err_Angle(deg)': 0.0, 'Ray_Geom/Err_Dist_L1(m)': 0.0, 'Ray_Geom/Endpoint_Offset_L2(m)': 0.0,
            'Ray_Geom/Err_Dist_Relative(%)': 0.0, 'Ray_Geom/Err_Dist_Short_<2.5m(m)': 0.0,
            'Ray_Geom/Err_Dist_Long_>=2.5m(m)': 0.0, 'Ray_Geom/Overshoot_>0.3m(%)': 0.0,
            'Ray_Geom/Undershoot_<-0.3m(%)': 0.0,
            'Refine_Geom/Err_Dist_Refined(m)': 0.0,
            'Refine_Geom/Dist_Improvement(m)': 0.0,
            'Refine_Geom_Final/Err_Endpoint_3D_Before(m)': 0.0,
            'Refine_Geom_Final/Err_Endpoint_3D_After(m)': 0.0,
            'Refine_Geom_Final/Perp_Dist_Before(m)': 0.0,
            'Refine_Geom_Final/Perp_Dist_After(m)': 0.0,
            'Refine_Geom_Final/Dir_Angle_Before(deg)': 0.0,
            'Refine_Geom_Final/Dir_Angle_After(deg)': 0.0,
            'Refine_Geom_Final/Offset_Magnitude(m)': 0.0,
            'Refine_Cls/Inside_Precision': 0.0,
            'Refine_Cls/Inside_Recall': 0.0,
            'Refine_Cls/Line_Precision': 0.0,
            'Refine_Cls/Line_Recall': 0.0,
            'Refine_Diag/Coverage_Rate(%)': 0.0,
            'Refine_Diag/Endpoint_Entropy': 0.0,
        })

        if pos_idx.shape[0] > 0 and gt_xyz_global_real.shape[0] > 0:
            ray_logits = model_output['ray_logits']
            ray_delta = model_output['ray_delta']
            ray_dist = model_output['ray_dist']
            flat_rays = gt['base_rays'].to(device)
            B = flat_rays.shape[0]

            dyn_gt_cls = model_output.get('dyn_gt_cls', None)
            dyn_gt_reg = model_output.get('dyn_gt_reg', None)

            if dyn_gt_cls is None:

                pred_xyz_matched_norm = model_output['corner_xyz_all'][pos_idx].detach()

                if model_output.get('sampled_gt_idx') is not None:
                    active_gt_idx = model_output['sampled_gt_idx'].long()
                else:
                    dists_active = torch.cdist(pred_xyz_matched_norm, gt_xyz_global_norm, p=2.0)
                    _, active_gt_idx = torch.min(dists_active, dim=1)

                dyn_gt_cls, dyn_gt_reg, _ = self._compute_dynamic_ray_targets(
                    pred_xyz=pred_xyz_matched_norm,
                    sampled_gt_idx=active_gt_idx,
                    gt_corner_xyz=gt_xyz_global_norm,
                    indptr=gt['corner_adj_indptr'].to(device).long(),
                    indices=gt['corner_adj_indices'].to(device).long(),
                    base_rays=flat_rays
                )

            gt_ray_cls = dyn_gt_cls
            gt_ray_reg = dyn_gt_reg

            probs = torch.sigmoid(ray_logits)
            preds = (probs > ray_threshold).float()
            targets = (gt_ray_cls > 0.5).float()

            preds_tp = preds[corner_tp_mask]
            targets_tp = targets[corner_tp_mask]

            tp = (preds_tp * targets_tp).sum()
            fp = (preds_tp * (1 - targets_tp)).sum()
            fn = ((1 - preds_tp) * targets_tp).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            metrics.update({
                'Ray_Stage1/Precision': precision.item(),
                'Ray_Stage1/Recall': recall.item(),
                'Ray_Stage1/F1': f1.item(),
                'Ray_Stage1/TP_Count': float(tp.item()),
            })

            train_refine_dict = model_output.get('train_refine_dict', None)
            if train_refine_dict is not None and 'line_targets' in train_refine_dict:
                train_logits = train_refine_dict['line_logits'].squeeze(-1)
                train_targets = train_refine_dict['line_targets']

                refine_probs = torch.sigmoid(train_logits)
                refine_preds = (refine_probs > 0.5).float()

                tp_train = (refine_preds * train_targets).sum()
                fp_train = (refine_preds * (1 - train_targets)).sum()
                fn_train = ((1 - refine_preds) * train_targets).sum()

                prec_train = tp_train / (tp_train + fp_train + 1e-8)
                rec_train = tp_train / (tp_train + fn_train + 1e-8)
                f1_train = 2 * prec_train * rec_train / (prec_train + rec_train + 1e-8)

                metrics.update({
                    'Ray_Refine_Train/Precision': prec_train.item(),
                    'Ray_Refine_Train/Recall': rec_train.item(),
                    'Ray_Refine_Train/F1': f1_train.item(),
                })

            if 'ray_refine_logits' in model_output and (model_output['ray_refine_logits'] > -100).any():
                eval_logits = model_output['ray_refine_logits']

                stage1_mask = (eval_logits > -100)
                stage2_mask = (torch.sigmoid(eval_logits) > 0.5)
                final_preds = (stage1_mask & stage2_mask).float()

                final_preds_tp = final_preds[corner_tp_mask]

                tp_final = (final_preds_tp * targets_tp).sum()
                fp_final = (final_preds_tp * (1 - targets_tp)).sum()
                fn_final = ((1 - final_preds_tp) * targets_tp).sum()

                prec_final = tp_final / (tp_final + fp_final + 1e-8)
                rec_final = tp_final / (tp_final + fn_final + 1e-8)
                f1_final = 2 * prec_final * rec_final / (prec_final + rec_final + 1e-8)

                metrics.update({
                    'Ray_Final_Cascade/Precision': prec_final.item(),
                    'Ray_Final_Cascade/Recall': rec_final.item(),
                    'Ray_Final_Cascade/F1': f1_final.item(),
                })

            corner_tp_mask_exp = corner_tp_mask.unsqueeze(1).expand(-1, B)
            mask_eval = (gt_ray_cls > 0.99) & corner_tp_mask_exp

            if mask_eval.any():
                p_dist_norm = ray_dist[mask_eval]
                t_dist_norm = gt_ray_reg[..., 3][mask_eval]

                if flat_rays.dim() == 2:
                    flat_rays_exp = flat_rays.unsqueeze(0).expand(ray_logits.shape[0], -1, -1)
                else:
                    flat_rays_exp = flat_rays

                active_base = flat_rays_exp[mask_eval]
                p_dir = F.normalize(active_base + ray_delta[mask_eval], dim=-1)
                t_delta = gt_ray_reg[..., 0:3][mask_eval]
                t_dir_abs = F.normalize(active_base + t_delta, dim=-1)

                cos_sim = F.cosine_similarity(p_dir, t_dir_abs, dim=-1).clamp(-1.0, 1.0)
                angle_deg = torch.acos(cos_sim) * (180.0 / math.pi)

                factor_val = n_factor.view(-1)[0].item() if isinstance(n_factor, torch.Tensor) else float(n_factor)
                p_dist_real = p_dist_norm / factor_val
                t_dist_real = t_dist_norm / factor_val

                abs_err_real = torch.abs(p_dist_real - t_dist_real)

                p_disp = p_dir * p_dist_real.unsqueeze(-1)
                t_disp = t_dir_abs * t_dist_real.unsqueeze(-1)
                endpoint_offset = torch.norm(p_disp - t_disp, dim=-1)

                short_mask = t_dist_real < 2.5
                long_mask = t_dist_real >= 2.5
                short_err = abs_err_real[short_mask].mean().item() if short_mask.any() else 0.0
                long_err = abs_err_real[long_mask].mean().item() if long_mask.any() else 0.0

                diff_real = p_dist_real - t_dist_real
                overshoot_rate = (diff_real > 0.3).float().mean().item() * 100.0
                undershoot_rate = (diff_real < -0.3).float().mean().item() * 100.0

                metrics.update({
                    'Ray_Geom/Err_Angle(deg)': angle_deg.mean().item(),
                    'Ray_Geom/Err_Dist_L1(m)': abs_err_real.mean().item(),
                    'Ray_Geom/Endpoint_Offset_L2(m)': endpoint_offset.mean().item(),
                    'Ray_Geom/Err_Dist_Relative(%)': (abs_err_real / (t_dist_real + 1e-3)).mean().item() * 100.0,
                    'Ray_Geom/Err_Dist_Short_<2.5m(m)': short_err,
                    'Ray_Geom/Err_Dist_Long_>=2.5m(m)': long_err,
                    'Ray_Geom/Overshoot_>0.3m(%)': overshoot_rate,
                    'Ray_Geom/Undershoot_<-0.3m(%)': undershoot_rate,
                })

            metrics.update({

                'Refine_Geom/Err_Dist_Refined(m)': 0.0,
                'Refine_Geom/Dist_Improvement(m)': 0.0,


                'Refine_Geom_Final/Err_Endpoint_3D_Before(m)': 0.0,  
                'Refine_Geom_Final/Err_Endpoint_3D_After(m)': 0.0,  
                'Refine_Geom_Final/Perp_Dist_Before(m)': 0.0,  
                'Refine_Geom_Final/Perp_Dist_After(m)': 0.0,  
                'Refine_Geom_Final/Dir_Angle_Before(deg)': 0.0,  
                'Refine_Geom_Final/Dir_Angle_After(deg)': 0.0,  
                'Refine_Geom_Final/Offset_Magnitude(m)': 0.0,  


                'Refine_Cls/Inside_Precision': 0.0,
                'Refine_Cls/Inside_Recall': 0.0,
                'Refine_Cls/Line_Precision': 0.0,
                'Refine_Cls/Line_Recall': 0.0,


                'Refine_Diag/Coverage_Rate(%)': 0.0,
                'Refine_Diag/Endpoint_Entropy': 0.0
            })

            train_refine_dict = model_output.get('train_refine_dict', None)

            if train_refine_dict is not None and 'line_targets' in train_refine_dict:
                inside_logits = train_refine_dict['support_logits']
                endpoint_logits = train_refine_dict['endpoint_logits']
                line_logits = train_refine_dict['line_logits']
                L_refined = train_refine_dict['L_refined']
                L_ext = train_refine_dict['L_ext']
                line_targets = train_refine_dict['line_targets']

                if 'delta_B_perp' in train_refine_dict:
                    delta_B_perp = train_refine_dict['delta_B_perp']
                else:
                    delta_B_perp = torch.zeros((inside_logits.shape[0], inside_logits.shape[1], 3), device=device)

                sampled_i = train_refine_dict['sampled_i']
                sampled_b = train_refine_dict['sampled_b']

                N_samples = inside_logits.shape[0]
                K_nodes = inside_logits.shape[1]

                if N_samples > 0:

                    line_preds = (torch.sigmoid(line_logits.squeeze(-1)) > 0.5).float()
                    tp_line = (line_preds * line_targets).sum()
                    fp_line = (line_preds * (1 - line_targets)).sum()
                    fn_line = ((1 - line_preds) * line_targets).sum()

                    prec_line = tp_line / (tp_line + fp_line + 1e-8)
                    rec_line = tp_line / (tp_line + fn_line + 1e-8)

                    metrics.update({
                        'Refine_Cls/Line_Precision': prec_line.item(),
                        'Refine_Cls/Line_Recall': rec_line.item()
                    })

                    pos_mask = (line_targets > 0.5)
                    if pos_mask.any():
                        pos_i = sampled_i[pos_mask]
                        pos_b = sampled_b[pos_mask]
                        gt_neighbor_idx = model_output.get('dyn_gt_endpoint', None)

                        if gt_neighbor_idx is not None:
                            gt_j = gt_neighbor_idx[pos_i, pos_b]
                            valid_gt = (gt_j >= 0)

                            if valid_gt.any():

                                q_gt = gt['corner_xyz'].to(device).float()[gt_j[valid_gt]]
                                A_pred = model_output['corner_xyz'][sampled_i]
                                dir_pred = F.normalize(flat_rays[sampled_b] + ray_delta[sampled_i, sampled_b],
                                                       dim=-1)

                                A_pos = A_pred[pos_mask][valid_gt]
                                dir_pos = dir_pred[pos_mask][valid_gt]
                                L_ext_pos = L_ext[pos_mask][valid_gt]

                                L_coarse_pos = ray_dist[pos_i[valid_gt], pos_b[valid_gt]]
                                L_refined_pos = L_refined[pos_mask][valid_gt]
                                delta_B_perp_pos = delta_B_perp[pos_mask][valid_gt]

                                v = q_gt - A_pos
                                t_gt = (v * dir_pos).sum(dim=-1)
                                closest = A_pos + t_gt.unsqueeze(-1) * dir_pos
                                perp_dist_before = torch.norm(closest - q_gt, dim=-1)

                                alphas = torch.linspace(0.0, 1.0, steps=K_nodes, device=device)
                                valid_node_dists = alphas.view(1, -1) * L_ext_pos.view(-1, 1)

                                inside_targets_hard = (valid_node_dists <= t_gt.unsqueeze(1)).float()

                                valid_inside_logits = inside_logits[pos_mask][valid_gt]
                                inside_preds = (torch.sigmoid(valid_inside_logits) > 0.5).float()

                                tp_ins = (inside_preds * inside_targets_hard).sum()
                                fp_ins = (inside_preds * (1 - inside_targets_hard)).sum()
                                fn_ins = ((1 - inside_preds) * inside_targets_hard).sum()

                                metrics.update({
                                    'Refine_Cls/Inside_Precision': (tp_ins / (tp_ins + fp_ins + 1e-8)).item(),
                                    'Refine_Cls/Inside_Recall': (tp_ins / (tp_ins + fn_ins + 1e-8)).item()
                                })

                                coverage_mask = (t_gt > 0) & (t_gt < L_ext_pos)
                                metrics[
                                    'Refine_Diag/Coverage_Rate(%)'] = coverage_mask.float().mean().item() * 100.0

                                factor_val = n_factor.view(-1)[0].item() if isinstance(n_factor,
                                                                                       torch.Tensor) else float(
                                    n_factor)

                                valid_endpoint_logits = endpoint_logits[pos_mask][valid_gt]
                                probs = F.softmax(valid_endpoint_logits, dim=-1)
                                metrics['Refine_Diag/Endpoint_Entropy'] = -(probs * torch.log(probs + 1e-8)).sum(
                                    dim=-1).mean().item()

                                valid_geom_mask = (t_gt > 0) & (perp_dist_before < 0.25 * factor_val)

                                if valid_geom_mask.any():
                                    t_gt_real = t_gt[valid_geom_mask] / factor_val
                                    L_refined_real = L_refined_pos[valid_geom_mask] / factor_val
                                    L_coarse_real = L_coarse_pos[valid_geom_mask] / factor_val

                                    refined_err_l1 = torch.abs(L_refined_real - t_gt_real).mean().item()
                                    improvement = torch.abs(
                                        L_coarse_real - t_gt_real).mean().item() - refined_err_l1

                                    metrics.update({
                                        'Refine_Geom/Err_Dist_Refined(m)': refined_err_l1,
                                        'Refine_Geom/Dist_Improvement(m)': improvement
                                    })

                                loose_mask = (t_gt > 0) & (perp_dist_before < 0.60 * factor_val)

                                if loose_mask.any():
                                    A_loose = A_pos[loose_mask]
                                    q_gt_loose = q_gt[loose_mask]
                                    dir_loose = dir_pos[loose_mask]

                                    L_ref_loose = L_refined_pos[loose_mask].unsqueeze(-1)
                                    delta_B_loose = delta_B_perp_pos[loose_mask]

                                    B_proj = A_loose + L_ref_loose * dir_loose
                                    err_3d_before = torch.norm(B_proj - q_gt_loose,
                                                               dim=-1).mean().item() / factor_val
                                    perp_before_m = perp_dist_before[loose_mask].mean().item() / factor_val

                                    B_final = B_proj + delta_B_loose
                                    err_3d_after = torch.norm(B_final - q_gt_loose,
                                                              dim=-1).mean().item() / factor_val

                                    v_final = q_gt_loose - A_loose
                                    t_final = (v_final * dir_loose).sum(dim=-1, keepdim=True)
                                    closest_final = A_loose + t_final * dir_loose

                                    perp_after_m = torch.norm(B_final - q_gt_loose,
                                                              dim=-1).mean().item() / factor_val

                                    dir_gt = F.normalize(q_gt_loose - A_loose, dim=-1)
                                    dir_final_pred = F.normalize(B_final - A_loose, dim=-1)

                                    cos_before = torch.sum(dir_loose * dir_gt, dim=-1).clamp(-1.0, 1.0)
                                    cos_after = torch.sum(dir_final_pred * dir_gt, dim=-1).clamp(-1.0, 1.0)

                                    angle_before = torch.acos(cos_before).mean().item() * (180.0 / math.pi)
                                    angle_after = torch.acos(cos_after).mean().item() * (180.0 / math.pi)

                                    offset_mag = torch.norm(delta_B_loose, dim=-1).mean().item() / factor_val

                                    metrics.update({
                                        'Refine_Geom_Final/Err_Endpoint_3D_Before(m)': err_3d_before,
                                        'Refine_Geom_Final/Err_Endpoint_3D_After(m)': err_3d_after,
                                        'Refine_Geom_Final/Perp_Dist_Before(m)': perp_before_m,
                                        'Refine_Geom_Final/Perp_Dist_After(m)': perp_after_m,
                                        'Refine_Geom_Final/Dir_Angle_Before(deg)': angle_before,
                                        'Refine_Geom_Final/Dir_Angle_After(deg)': angle_after,
                                        'Refine_Geom_Final/Offset_Magnitude(m)': offset_mag
                                    })
            return metrics
        return metrics

    @torch.no_grad()
    def val_cube_pred(self, out_cube, gt_cube):
        pred = out_cube.view_as(gt_cube).to(torch.bool)
        gt   = gt_cube.to(torch.bool)
        tp = (pred &  gt).sum().float()
        fp = (pred & ~gt).sum().float()
        fn = (~pred &  gt).sum().float()
        tn = (~pred & ~gt).sum().float()
        pos_pred = (tp + fp).clamp_min(1.0)
        pos_gt   = (tp + fn).clamp_min(1.0)
        union    = (tp + fp + fn).clamp_min(1.0)
        total    = (tp + fp + fn + tn).clamp_min(1.0)
        precision = tp / pos_pred
        recall    = tp / pos_gt
        f1_denom  = (precision + recall)
        f1        = (2 * precision * recall / f1_denom) if f1_denom > 0 else torch.tensor(0.0, device=precision.device)
        iou       = tp / union
        accuracy  = (tp + tn) / total
        return recall.item(), precision.item(), f1.item(), iou.item(), accuracy.item()
