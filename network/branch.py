import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp import MLP

from SelfAttention import SelfAttnBlock
from CrossAttention import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from pytorch3d.ops import knn_points
import math


class LineRefineNetwork(nn.Module):

    def __init__(self, p):
        super(LineRefineNetwork, self).__init__()

        self.num_nodes = p.num_nodes  
        self.top_k = p.top_k  
        self.tau = p.tau  

        self.pos_encoder = MLP(**p.pos_encoder)

        self.feat_proj = MLP(**p.feat_proj)

        self.node_mixer = nn.Sequential(
            nn.Conv1d(p.conv_in, p.conv_in, kernel_size=3, padding=1),
            nn.LayerNorm([p.conv_in, self.num_nodes]),
            nn.ReLU(inplace=True),
            nn.Conv1d(p.conv_in, p.conv_out, kernel_size=3, padding=1),
            nn.LayerNorm([p.conv_out, self.num_nodes]),
            nn.ReLU(inplace=True)
        )

        self.support_head = MLP(**p.support_head)

        self.endpoint_head = MLP(**p.endpoint_head)

        self.line_cls_head = nn.Sequential(
            MLP(**p.line_cls_head1),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),  
            MLP(**p.line_cls_head2)
        )

        self.offset_head = nn.Sequential(
            MLP(**p.offset_head1),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),  
            MLP(**p.offset_head2),  
        )

    def forward(self, A, pred_dir, L_coarse, edge_xyz, edge_feat, L_ext_override=None):

        N = A.shape[0]
        E = edge_xyz.shape[0]
        K = self.num_nodes

        if N == 0 or E == 0:
            return {
                'support_logits': torch.zeros((0, K), device=A.device),
                'endpoint_logits': torch.zeros((0, K), device=A.device),
                'line_logits': torch.zeros((0, 1), device=A.device),
                'L_refined': torch.zeros((0,), device=A.device),
                'delta_B_perp': torch.zeros((0, 3), device=A.device),
                'L_ext': torch.zeros((0,), device=A.device),
            }

        if L_ext_override is not None:
            L_ext = L_ext_override  
        else:
            L_ext = torch.clamp(1.5 * L_coarse, max=2.0)  

        alphas = torch.linspace(0.0, 1.0, steps=K, device=A.device)  
        node_dists = alphas.view(1, K) * L_ext.view(N, 1)  

        nodes = A.unsqueeze(1) + pred_dir.unsqueeze(1) * node_dists.unsqueeze(-1)

        nodes_flat = nodes.reshape(-1, 3)  

        actual_k = min(self.top_k, E)

        with torch.no_grad():
            knn_res = knn_points(
                nodes_flat.unsqueeze(0),  
                edge_xyz.unsqueeze(0),  
                K=actual_k,
                return_nn=False,
            )

            topk_dists = torch.sqrt(knn_res.dists.squeeze(0).clamp(min=1e-8))
            topk_idx = knn_res.idx.squeeze(0)

            weights = F.softmax(-topk_dists / self.tau, dim=1)  

            k_feats = edge_feat[topk_idx]  
            k_xyz = edge_xyz[topk_idx]  

            nearest_feat = torch.sum(weights.unsqueeze(-1) * k_feats, dim=1)
            nearest_xyz_ref = torch.sum(weights.unsqueeze(-1) * k_xyz, dim=1)

        nearest_feat = nearest_feat.reshape(N, K, -1)  
        nearest_xyz_ref = nearest_xyz_ref.reshape(N, K, 3)  

        rel_pos = nearest_xyz_ref - nodes  

        alpha_expand = alphas.view(1, K, 1).expand(N, K, 1)  
        L_coarse_expand = L_coarse.view(N, 1, 1).expand(N, K, 1)  
        dir_expand = pred_dir.view(N, 1, 3).expand(N, K, 3)  

        geo_info = torch.cat(
            [rel_pos, dir_expand, alpha_expand, L_coarse_expand],
            dim=-1,
        )

        pos_encoded = self.pos_encoder(geo_info)  

        fused_input = torch.cat([nearest_feat, pos_encoded], dim=-1)

        node_feats = self.feat_proj(fused_input)

        node_feats = node_feats.permute(0, 2, 1).contiguous()  

        mixed_feats = self.node_mixer(node_feats)  
        mixed_feats = mixed_feats.permute(0, 2, 1).contiguous()  

        support_logits = self.support_head(mixed_feats).squeeze(-1)  
        endpoint_logits = self.endpoint_head(mixed_feats).squeeze(-1)  

        p_end = F.softmax(endpoint_logits, dim=-1)  

        L_refined = torch.sum(p_end * node_dists, dim=-1)  

        max_feat = mixed_feats.max(dim=1)[0]  
        mean_feat = mixed_feats.mean(dim=1)  
        end_feat = torch.sum(p_end.unsqueeze(-1) * mixed_feats, dim=1)  

        global_feat = torch.cat([max_feat, mean_feat, end_feat], dim=-1)  

        line_logits = self.line_cls_head(global_feat)  

        offset_raw = self.offset_head(global_feat)  

        dir_component = (offset_raw * pred_dir).sum(dim=-1, keepdim=True) * pred_dir
        offset_perp = offset_raw - dir_component  

        max_angle_rad = 10.0 * (math.pi / 180.0)
        base_L = torch.clamp_min(L_refined.detach(), 0.1)  
        max_perp_allow = base_L * math.tan(max_angle_rad)  

        norm_perp = torch.norm(offset_perp, dim=-1, keepdim=True).clamp_min(1e-8)
        offset_dir = offset_perp / norm_perp  

        delta_B_perp = (
                offset_dir
                * max_perp_allow.unsqueeze(-1)
                * torch.tanh(norm_perp)
        )

        return {
            'support_logits': support_logits,  
            'endpoint_logits': endpoint_logits,  
            'line_logits': line_logits,  
            'L_refined': L_refined,  
            'delta_B_perp': delta_B_perp,  
            'L_ext': L_ext,  
        }


class JointCornerRayDecoderSimple(nn.Module):
    def __init__(self, p):
        super(JointCornerRayDecoderSimple, self).__init__()

        self.mlp_center_mul = MLP(**p.mlp_center_mul)

        self.mlp_center_add = MLP(**p.mlp_center_add)

        self.mlp_gate = MLP(**p.mlp_gate)

        self.decoder2corner = MLP(**p.decoder2corner)  
        self.max_corner_offset = p.max_corner_offset

        self.delta_embed = MLP(**p.delta_embed)

        self.corner_delta_fuse = MLP(**p.corner_delta_fuse)

        self.offset_scale = p.offset_scale

        self.edge_enc = MLP(**p.edge_enc)
        self.edge_feat_enc = MLP(**p.edge_feat_enc)
        self.edge_fuse = MLP(**p.edge_fuse)
        self.ca_block = CrossAttnBlock2D(p.ca)

        self.ray_pos_encoder = MLP(**p.ray_pos_encoder)

        self.ray_sa = SelfAttnBlock(p.sa)

        self.scale = p.ray_scale_dim ** -0.5

        self.reg_head = MLP(**p.reg_head)

    def forward(self, center_xyz, center_feat, token_context, edge_xyz, edge_feat, flat_rays, gt_corner_xyz=None,
                mode='train'):

        c_mul = self.mlp_center_mul(center_feat)
        c_add = self.mlp_center_add(center_feat)

        gate = 1.0 + 0.5 * torch.tanh(self.mlp_gate(token_context))
        corner_query = c_mul * gate + c_add  

        raw_delta = self.decoder2corner(corner_query)  
        delta = self.max_corner_offset * torch.tanh(raw_delta)
        corner_pred_xyz_all = center_xyz + delta  

        delta_feat = self.delta_embed(delta)
        corner_feat_all = corner_query + self.corner_delta_fuse(
            torch.cat([corner_query, c_add, delta_feat], dim=-1)
        )

        if mode == 'train':

            with torch.no_grad():
                dist_mat = torch.cdist(gt_corner_xyz, corner_pred_xyz_all, p=2.0)
                pred2gt_min_dist, _ = torch.min(dist_mat, dim=0)

                cost_matrix = dist_mat.cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                sampled_gt_idx = torch.tensor(row_ind, dtype=torch.long, device=gt_corner_xyz.device)
                pos_idx = torch.tensor(col_ind, dtype=torch.long, device=gt_corner_xyz.device)

            tb_pos = corner_feat_all.index_select(0, pos_idx)
            corner_offset_pos = corner_pred_xyz_all.index_select(0, pos_idx)

        else:

            pred2gt_min_dist = None
            sampled_gt_idx = None

            pos_idx = torch.arange(corner_feat_all.shape[0], dtype=torch.long, device=corner_feat_all.device)
            tb_pos = corner_feat_all.index_select(0, pos_idx)
            corner_offset_pos = corner_pred_xyz_all.index_select(0, pos_idx)

        if tb_pos.shape[0] > 0:
            e_pos = self.edge_enc(edge_xyz)
            e_sem = self.edge_feat_enc(edge_feat)
            kv_edge = self.edge_fuse(torch.cat([e_pos, e_sem], dim=-1))

            q_corner = self.ca_block(tb_pos, kv_edge)

            ray_feat = self.ray_pos_encoder(flat_rays)
            ray_feat = self.ray_sa(ray_feat)
            ray_logits = torch.matmul(q_corner, ray_feat.t()) * self.scale

            S_num, Bins_num = ray_logits.shape
            q_expanded = q_corner.unsqueeze(1).expand(-1, Bins_num, -1)
            ray_expanded = ray_feat.unsqueeze(0).expand(S_num, -1, -1)

            sparse_input = torch.cat([q_expanded, ray_expanded], dim=-1)
            ray_reg = self.reg_head(sparse_input)

            ray_delta = torch.tanh(ray_reg[..., 0:3]) * self.offset_scale
            fine_dist = F.softplus(ray_reg[..., 3])
        else:

            ray_logits = corner_query.new_zeros((0, flat_rays.shape[0]))
            ray_delta = corner_query.new_zeros((0, flat_rays.shape[0], 3))
            fine_dist = corner_query.new_zeros((0, flat_rays.shape[0]))

        return {

            'corner_xyz_all': corner_pred_xyz_all,
            'pred2gt_min_dist': pred2gt_min_dist,
            'corner_offset_pos': corner_offset_pos,
            'ray_logits': ray_logits,
            'ray_delta': ray_delta,
            'ray_dist': fine_dist,
            'active_idx': pos_idx,
            'sampled_gt_idx': sampled_gt_idx,
        }


class EdgeDecoder(nn.Module):
    def __init__(self, p):
        super(EdgeDecoder, self).__init__()
        self.mlp_gate = MLP(**p.mlp_gate)
        self.decoder2edge = MLP(**p.mlp_edge)
        self.ln_after_cat = nn.LayerNorm(256)

    def forward(self, point_feat, token_context, fp_idx, fp_dist):

        fp_idx = fp_idx.long()
        fp_dist = fp_dist.float()

        fp_context = token_context[fp_idx]  

        weight = 1.0 / (fp_dist + 1e-8)  
        weight = weight / torch.sum(weight, dim=1, keepdim=True)  

        context_per_point = torch.sum(fp_context * weight.unsqueeze(-1), dim=1)  

        gate_raw = self.mlp_gate(context_per_point)  

        gate = 1.0 + 0.5 * torch.tanh(gate_raw)

        edge_feat = point_feat * gate  

        edge_res = self.decoder2edge(edge_feat)

        return edge_feat, edge_res


class PointSimpleEncoder(nn.Module):
    def __init__(self, p):
        super(PointSimpleEncoder, self).__init__()

        self.mlp_abs = MLP(**p.mlp_abs)

        if p.max_pooling:
            self.use_max_pooling = True
        else:
            self.use_max_pooling = False
            self.knn_feat = nn.Sequential(
                nn.Conv1d(p.N_knn, p.N_knn, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(p.N_knn, 1, 1)
            )

        self.mlp_self = MLP(**p.mlp_self)

        self.mlp_block = MLP(**p.mlp_block)

        self.mlp_token = MLP(**p.mlp_token)

        self.self_attn_block = SelfAttnBlock(p.attn)

    def forward(self, model_input):
        points = model_input['pc'].float()  
        knn_idx = model_input['pc_KNN_idx'].long()  

        pc_KNN_pos = points[knn_idx] - points.unsqueeze(1)
        feat = self.mlp_abs(pc_KNN_pos)  
        if self.use_max_pooling:
            feat = torch.max(feat, dim=1, keepdim=True)[0]  
        else:
            feat = self.knn_feat(feat)  
        m_abs = feat.squeeze(1)  

        m_self = self.mlp_self(points)  

        m_hyb = torch.cat([m_abs, m_self], dim=-1)  

        block_idx = model_input['block_idx'].long()  
        centers_idx = model_input['centers_idx'].long()  

        center_xyz = points[centers_idx]  
        center_feat = m_hyb[centers_idx]  

        group_xyz = points[block_idx]  
        group_feat = m_hyb[block_idx]  

        rel_xyz = group_xyz - center_xyz.unsqueeze(1)  
        dist = torch.norm(rel_xyz, dim=-1, keepdim=True)  
        local_input = torch.cat([group_feat, rel_xyz, dist], dim=-1)  

        local_feat = self.mlp_block(local_input)

        token_max = torch.max(local_feat, dim=1)[0]  
        token_mean = torch.mean(local_feat, dim=1)  

        token_feat = torch.cat([token_max, token_mean, center_feat], dim=-1)  

        token_feat = self.mlp_token(token_feat)  

        token_context = self.self_attn_block(token_feat)  

        return token_context, m_self, m_hyb


if __name__ == '__main__':
    from dotted.collection import DottedDict

    params = DottedDict({
        'max_pooling': True,
        'grid_size': 8,
        'mlp': {
            'size': [3, 128, 128],
            'activation_type': 'lrelu',
            'num_pos_encoding': -1
        },
        'grid_conv': {
            'latent_size': 128,
            'conv_dim': 3,
            'num_conv': 3,
            'activation': 'lrelu',
            'kernel_size': 3,
            'padding': 1,
        }
    })

    points = torch.rand(10, 3) * 2 - 1
    model_input = {
        'pc_KNN_pos': torch.rand((10, 4, 3)),
        'points': points
    }
