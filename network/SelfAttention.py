import numpy as np
import torch
from torch import nn
from torch.nn import init

from mlp import MLP

class Block(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.h = int(p.head)
        self.head_dim = int(p.head_dim)
        self.d_model = self.h * self.head_dim
        self.activation_type = p.activation_type
        self.ffn_ratio = float(p.ffn_ratio)         
        self.drop = nn.Dropout(float(p.dropout))    

        self.ln1 = nn.LayerNorm(self.d_model)
        self.qkv_mlp = MLP(
            size=[self.d_model, self.d_model, 3 * self.d_model],
            activation_type=self.activation_type,
            bias=True, num_pos_encoding=-1
        )
        self.proj = nn.Linear(self.d_model, self.d_model)

        self.ffn = None
        if self.ffn_ratio > 0.0:
            hidden = int(self.d_model * self.ffn_ratio)
            self.ln2 = nn.LayerNorm(self.d_model)
            self.ffn = MLP(
                size=[self.d_model, hidden, self.d_model],
                activation_type=self.activation_type,
                bias=True,  num_pos_encoding=-1
            )

        self.rel_scale_raw = nn.Parameter(torch.tensor(2.0))  

    def _mhsa(self, q, k, v, rel_bias=None):
        d = q.shape[-1]
        att = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)   
        if rel_bias is not None:
            rel_scale = torch.nn.functional.softplus(self.rel_scale_raw)           

            att = att + rel_scale * rel_bias
        att = torch.softmax(att, dim=-1)
        ctx = torch.matmul(att, v)                                 
        return ctx

    def forward(self, x, rel_bias=None):
        y = self.ln1(x)                                
        qkv = self.qkv_mlp(y)           
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        h, d = self.h, self.head_dim
        q = q.view(-1, h, d).transpose(0, 1).contiguous()  
        k = k.view(-1, h, d).transpose(0, 1).contiguous()
        v = v.view(-1, h, d).transpose(0, 1).contiguous()

        ctx = self._mhsa(q, k, v, rel_bias=rel_bias)     
        ctx = ctx.transpose(0, 1).contiguous().view(-1, self.d_model)  
        ctx = self.proj(ctx)
        ctx = self.drop(ctx)
        x = x + ctx

        if self.ffn is not None:
            y2 = self.ln2(x)
            ff = self.ffn.forward_simple(y2)
            ff = self.drop(ff)
            x  = x + ff
        return x

class SelfAttnBlock(nn.Module):

    def __init__(self, p):
        super().__init__()

        depth = int(p.depth)
        self.blocks = nn.ModuleList([Block(p.block) for _ in range(depth)])

    def forward(self, x, pos: torch.Tensor = None, rel_bias: torch.Tensor = None):

        for blk in self.blocks:
            if pos is None:
                x = blk(x, rel_bias=rel_bias)
            else:
                x = blk(x + pos, rel_bias=rel_bias)
        return x
