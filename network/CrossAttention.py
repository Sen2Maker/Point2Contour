import numpy as np
import torch
from torch import nn
from torch.nn import init

from mlp import MLP


class CrossBlock2D(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.h = int(p.head)
        self.head_dim = int(p.head_dim)
        self.d_model = self.h * self.head_dim

        self.ln_q = nn.LayerNorm(self.d_model)
        self.ln_kv = nn.LayerNorm(self.d_model)

        self.q_mlp = MLP(size=[self.d_model, self.d_model, self.d_model],
                         activation_type=p.activation_type, bias=True)
        self.kv_mlp = MLP(size=[self.d_model, self.d_model, 2 * self.d_model],
                          activation_type=p.activation_type, bias=True)

        self.proj = nn.Linear(self.d_model, self.d_model)
        self.drop = nn.Dropout(float(p.dropout))

        self.ffn_ratio = float(p.ffn_ratio)
        self.ffn = None
        if self.ffn_ratio > 0.0:
            hidden = int(self.d_model * self.ffn_ratio)
            self.ln2 = nn.LayerNorm(self.d_model)
            self.ffn = MLP(size=[self.d_model, hidden, self.d_model],
                           activation_type=p.activation_type, bias=True)

    def _mhsa(self, q, k, v):

        d = q.shape[-1]
        att = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)  
        att = torch.softmax(att, dim=-1)
        ctx = torch.matmul(att, v)  
        return ctx

    def forward(self, x, context):

        N, C = x.shape
        M, _ = context.shape
        residual = x

        x_norm = self.ln_q(x)
        ctx_norm = self.ln_kv(context)

        q = self.q_mlp(x_norm)  
        kv = self.kv_mlp(ctx_norm)  
        k, v = torch.chunk(kv, 2, dim=-1)

        q = q.view(N, self.h, self.head_dim).permute(1, 0, 2)
        k = k.view(M, self.h, self.head_dim).permute(1, 0, 2)
        v = v.view(M, self.h, self.head_dim).permute(1, 0, 2)

        ctx = self._mhsa(q, k, v)

        ctx = ctx.permute(1, 0, 2).reshape(N, C)
        ctx = self.proj(ctx)
        ctx = self.drop(ctx)
        x = residual + ctx

        if self.ffn is not None:
            residual = x
            y2 = self.ln2(x)
            ff = self.ffn(y2)
            ff = self.drop(ff)
            x = residual + ff
        return x


class CrossAttnBlock2D(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.blocks = nn.ModuleList([CrossBlock2D(p.block) for _ in range(p.depth)])

    def forward(self, x, context):
        for blk in self.blocks:
            x = blk(x, context)
        return x
