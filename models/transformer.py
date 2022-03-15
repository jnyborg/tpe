import torch
import torch.nn as nn
import numpy as np
import math

from models.layers import LinearLayer, get_positional_encoding
from models.ltae import FourierPositionalEncoding

import torch
from torch import nn

from einops import rearrange, repeat
import torch.nn.functional as F



class Transformer(nn.Module):
    # Transformer in the style of the LTAE
    # - no value embedding
    # - simplified out embedding
    def __init__(self, in_channels=128, n_head=16, d_k=8,
            dropout=0.2, d_model=256, T=1000, return_att=False,
            with_pos_enc=True, with_gdd_enc=False, pos_type='default', pool_type='max',
            max_temporal_shift=60):

        super(Transformer, self).__init__()
        self.d_model = d_model
        self.with_pos_enc = with_pos_enc

        self.with_gdd_enc = with_gdd_enc
        self.pos_type = pos_type
        self.max_temporal_shift = max_temporal_shift
        if with_pos_enc:
            if pos_type == 'default':
                if with_gdd_enc:
                    sin_tab = get_positional_encoding(10000, self.d_model, T=10000)
                    self.position_enc = nn.Embedding.from_pretrained(sin_tab, freeze=True)
                else:
                    sin_tab = get_positional_encoding(365 + 2*max_temporal_shift, self.d_model, T=T)
                    self.position_enc = nn.Embedding.from_pretrained(sin_tab, freeze=True)
            elif pos_type == 'fourier':
                self.position_enc = FourierPositionalEncoding(m=1, f=128, h=32, d=d_model, groups=1)
            else:
                self.position_enc = None
        else:
            self.position_enc = None

        self.in_emb = LinearLayer(in_channels, d_model)
        self.attn = SelfAttention(d_model, heads=n_head, dim_head=d_k, dropout=0.1)
        self.ff = nn.Sequential(
                LinearLayer(d_model, 128),
                nn.Dropout(0.1))

        self.pool_type = pool_type
        if self.pool_type == 'attention':
            self.attention_pool = nn.Linear(128, 1, bias=False)
        elif self.pool_type == 'ltae':
            self.ltae = LTAE(n_head, d_k, d_model, dropout=0.1)
        elif self.pool_type == 'max':
            pass
        elif self.pool_type == 'last':
            pass
        else:
            raise NotImplementedError()





    def forward(self, x, positions, gdd):
        b, t, _ = x.shape
        x = self.in_emb(x)

        if self.position_enc is not None:
            if self.pos_type == 'default':
                if self.with_gdd_enc:
                    x = x + self.position_enc((gdd * 10000).long())
                else:
                    x = x + self.position_enc(positions + self.max_temporal_shift)
            elif self.pos_type == 'fourier':
                x = x + self.position_enc(gdd)

        x = self.attn(x)
        x = self.ff(x)

        # classification type
        if self.pool_type == 'last':
            x = x[:, -1]  # classify last time step
        elif self.pool_type == 'attention':  # even simpler LTAE
            # (b t c) -> (b t 1) -> (b 1 t) -> (b 1 t) @ (b t c) -> (b c)
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        elif self.pool_type == 'ltae':
            x = self.ltae(x)
        elif self.pool_type == 'max':
            x, _ = torch.max(x, dim=1)
        else:
            x = x.mean(dim=1)  # average over time (GAP?)


        return x

class AttentionQuery(nn.Module):
    def __init__(self, n_head, d_k, d_in, t_out=1, dropout=0.):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.t_out = t_out

        self.key = nn.Linear(d_in, n_head * d_k, bias=False)
        self.query = nn.Linear(d_in, 1, bias=False)

        self.temperature = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        B, T, C = x.size()

        q = self.query.repeat(B, 1, 1, 1).transpose(1, 2)  # (nh, hs) -> (B, nh, t_out, d_k)

        k = self.key(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)  # (B, nh, T, d_k)
        v = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # self-attend; (B, nh, t_out, d_k) x (B, nh, d_k, T) -> (B, nh, t_out, T)
        att = (q @ k.transpose(-2, -1)) / self.temperature
        att = self.softmax(att)
        att = self.dropout(att)
        y = att @ v  # (B, nh, t_out, T) x (B, nh, T, hs) -> (B, nh, t_out, hs)
        if self.t_out != 1:
            y = y.transpose(1, 2).contiguous().view(B, self.t_out, C)
        else:
            y = y.transpose(1, 2).contiguous().view(B, C)
        return y



class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Identity()
        self.dropout = nn.Dropout(dropout)



    def forward(self, x):
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        qkv = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


class AttentionPositionalBias(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Identity()
        self.dropout = nn.Dropout(dropout)

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
            else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
 
        self.slopes = torch.Tensor(get_slopes(heads)).cuda()
        #maxpos = 365
        ##In the next line, the part after the * is what constructs the diagonal matrix (right matrix in Figure 3 in the paper). 
        ##If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3, but one where all rows are identical.
        ##This works because the softmax operation is invariant to translation, and our bias functions are always linear. 
        #self.alibi = torch.arange(maxpos).unsqueeze(0).unsqueeze(0).expand(heads, -1, -1)
        #print(self.alibi)
        #self.alibi = self.alibi * self.slopes.unsqueeze(1).unsqueeze(1)
        #self.alibi = self.alibi.view(heads, 1, maxpos)
        #self.alibi = self.alibi.repeat(128, 1, 1)  # batch_size, 1, 1

        #print(self.alibi)
        #exit()

        # self.pos_enc = nn.Embedding.from_pretrained(
        #     get_positional_encoding(365*2, heads, T=10000), freeze=True)

        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()


    def forward(self, x, positions):
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        qkv = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # (b nh t t)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # relative positional encoding
        positions = repeat(positions, 'b t -> b t t2', t2=positions.shape[1])
        diag = torch.diagonal(positions, dim1=1, dim2=2).unsqueeze(1)
        # non-symmetric
        # relative_positions = (diag - positions)

        # symmetric
        relative_positions = -torch.abs(diag - positions)
        relative_positions = repeat(relative_positions, 'b t1 t2 -> b h t1 t2', h=self.heads)
        alibi = self.slopes[None, :, None, None] * relative_positions

        dots = dots + alibi

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out
