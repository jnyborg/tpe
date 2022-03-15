"""

Lightweight Temporal Attention Encoder module

Credits:
The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
the Transformer served as code base for the present script.

paper: https://arxiv.org/abs/1706.03762
code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""

from models.layers import LinearLayer
import torch
import torch.nn as nn
import numpy as np
import math
import copy
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops.einops import rearrange, repeat


class LTAE(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128],
            dropout=0.2, d_model=256, T=1000, return_att=False,
            with_pos_enc=True, with_gdd_pos=False, pos_type='default',
            max_temporal_shift=60):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)

        """

        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.n_neurons = copy.deepcopy(n_neurons)
        self.return_att = return_att


        if d_model is not None:
            self.d_model = d_model
            # self.inconv = nn.Linear(in_channels, d_model)
            self.inconv = LinearLayer(in_channels, d_model)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.max_temporal_shift = max_temporal_shift
        self.pos_type = pos_type
        self.with_gdd_pos = with_gdd_pos
        self.n_head = n_head
        if with_pos_enc:
            if pos_type == 'default':
                if with_gdd_pos:
                    sin_tab = get_positional_encoding(10000, self.d_model // n_head, T=10000)
                    self.position_enc = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
                                                                     freeze=True)
                else:
                    sin_tab = get_positional_encoding(365 + 2*max_temporal_shift, self.d_model // n_head, T=T)
                    self.position_enc = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
                                                                     freeze=True)
            elif self.pos_type == 'fourier':
                dim = self.d_model // n_head
                if with_gdd_pos:
                    self.position_enc = LearnableFourierPositionalEncoding(m=1, f=dim, h=32, d=dim, max_pos=10000, n_head=n_head)
                else:
                    self.position_enc = LearnableFourierPositionalEncoding(m=1, f=dim, h=32, d=dim, max_pos=365 + 2*max_temporal_shift, n_head=n_head)
            elif self.pos_type == 'rnn':
                self.position_enc = RNNPositionalEncoding(d_model, n_head, sinusoid=True)
            else:
                self.position_enc = None
        else:
            self.position_enc = None

        # self.inlayernorm = nn.LayerNorm(self.in_channels)
        # self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model)
        assert (self.n_neurons[0] == self.d_model)

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.append(LinearLayer(self.n_neurons[i], self.n_neurons[i + 1]))
        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions, gdd):
        # x = self.inlayernorm(x)

        if self.inconv is not None:
            x = self.inconv(x)

        if self.with_gdd_pos:
            positions = gdd
        else:
            positions = positions + self.max_temporal_shift  # for ShiftAug


        if self.position_enc is not None:
            x = x + self.position_enc(positions)

        x, attn = self.attention_heads(x)


        x = self.dropout(self.mlp(x))
        # x = self.outlayernorm(x)

        if self.return_att:
            return x, attn
        else:
            return x

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_k, d_in, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.key = nn.Linear(d_in, n_head * d_k)
        self.query = nn.Parameter(torch.zeros(n_head, d_k)).requires_grad_(True)
        nn.init.normal_(self.query, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.temperature = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, key=None):
        B, T, C = x.size()

        q = self.query.repeat(B, 1, 1, 1).transpose(1, 2)  # (nh, hs) -> (B, nh, 1, d_k)
        k = self.key(x if key is None else key).view(B, T, self.n_head, self.d_k).transpose(1, 2)  # (B, nh, T, d_k)
        v = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # self-attend; (B, nh, 1, d_k) x (B, nh, d_k, T) -> (B, nh, 1, T)
        att = (q @ k.transpose(-2, -1)) / self.temperature
        att = self.softmax(att)
        att = self.dropout(att)
        y = att @ v  # (B, nh, 1, T) x (B, nh, T, hs) -> (B, nh, 1, hs)
        y = y.transpose(1, 2).contiguous().view(B, C)
        return y, att


def get_positional_encoding(max_len, d_model, T=1000.0):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(T) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, m=1, f=384, h=32, d=768, max_pos=10000, n_head=16):
        """
        Re-implementation of Learnable Fourier Features from https://arxiv.org/abs/2106.02795
        """

        super().__init__()

        assert f % 2 == 0

        self.wr = nn.Linear(m, f//2, bias=False)
        self.max_pos = max_pos

        self.mlp = nn.Sequential(
                nn.Linear(f, h),
                nn.GELU(),
                nn.Linear(h, d)
        )
        self.scale = f**-0.5
        self.n_head = n_head

    def forward(self, x):
        x = x.unsqueeze(2).float() / self.max_pos  # normalize to [0, 1]

        x = self.wr(x)
        x = self.scale * torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        x = self.mlp(x)
        x = torch.cat([x for _ in range(self.n_head)], dim=2)

        return x



class RNNPositionalEncoding(nn.Module):
    def __init__(self, d_model, n_head, sinusoid=True, max_pos=10000):
        super().__init__()
        dim = d_model // n_head
        self.sinusoid = sinusoid
        if self.sinusoid:
            sin_tab = get_positional_encoding(max_pos, dim, T=10000)
            self.position_enc = nn.Embedding.from_pretrained(sin_tab, freeze=True)
            input_dim = dim
        else:
            input_dim = 1

        self.rnn = nn.GRU(input_dim, dim, batch_first=True)
        self.mlp = nn.Linear(dim, dim)
        self.n_head = n_head
        self.max_pos = max_pos


    def forward(self, x):
        if not self.sinusoid:
            x = x.unsqueeze(2) / self.max_pos  # normalize to [0, 1]
        else:
            x = self.position_enc(x)
        x, _ = self.rnn(x)
        x = self.mlp(x)
        x = torch.cat([x for _ in range(self.n_head)], dim=2)
        return x
