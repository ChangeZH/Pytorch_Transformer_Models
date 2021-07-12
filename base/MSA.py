import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange


class MSA(nn.Module):
    """docstring for MSA"""

    def __init__(self, dim, inter_dim, head, dropout_ratio):
        super(MSA, self).__init__()
        self.dim = dim
        self.inter_dim = inter_dim
        self.head = head
        self.scale = self.inter_dim ** -.5
        self.fc_qkv = nn.Linear(dim, self.inter_dim * 3, bias=False)
        self.softmax = nn.Softmax(-1)
        self.fc_out = nn.Sequential(
            nn.Linear(self.inter_dim, dim, bias=False),
            nn.Dropout(dropout_ratio)
        )

    def softmax_with_mask(self, feats, mask):
        feats_exp = torch.where(mask, feats.exp(), torch.zeros_like(feats))
        feats_exp_sum = torch.sum(feats_exp, dim=-1, keepdim=True).expand_as(feats_exp)
        return feats_exp / torch.where(feats_exp_sum == 0, torch.ones_like(feats_exp_sum), feats_exp_sum)

    def forward(self, features, mask=None):
        q, k, v = self.fc_qkv(features).chunk(3, dim=-1)
        q, k, v = rearrange(q, 'b n (h d) -> b n h d', h=self.head), \
                  rearrange(k, 'b n (h d) -> b n h d', h=self.head), \
                  rearrange(v, 'b n (h d) -> b n h d', h=self.head)
        alpha = einsum('b i h d, b j h d -> b h i j', q, k) * self.scale
        if mask is None:
            att = self.softmax(alpha)
            out = einsum('b h i j, b j h d -> b i h d', att, v)
            out = rearrange(out, 'b n h d -> b n (h d)')
        else:
            out = []
            ids = torch.unique(mask)
            for id in ids:
                att = self.softmax_with_mask(alpha, (mask == id).unsqueeze(1).expand_as(alpha))
                out.append(rearrange(einsum('b h i j, b j h d -> b i h d', att, v), 'b n h d -> b n (h d)'))
            out = sum(out)
        return self.fc_out(out)


class EMSA(nn.Module):
    """docstring for EMSA"""

    def __init__(self, input_size, stage, dim, head, dropout_ratio):
        super(EMSA, self).__init__()
        self.input_size = input_size
        self.stage = stage
        self.dim = dim
        self.head = head
        self.scale = dim ** -.5
        self.trans_q = nn.Linear(dim, dim, bias=False)
        self.trans_kv = nn.Sequential(Rearrange('b (w h) c -> b c w h', w=int(self.input_size // 2 ** (self.stage + 2)),
                                                h=int(self.input_size // 2 ** (self.stage + 2))),
                                      nn.Conv2d(in_channels=self.dim, out_channels=2 * self.dim,
                                                kernel_size=3, stride=2, padding=1),
                                      Rearrange('b c w h -> b (w h) c'),
                                      nn.LayerNorm(normalized_shape=2 * self.dim))

        self.num_features = (self.input_size / (2 ** (self.stage + 2))) ** 2 * (
                self.input_size / (2 ** (self.stage + 3))) ** 2 if self.input_size / (
                2 ** (self.stage + 3)) % 4 == 0 else (self.input_size / (2 ** (self.stage + 2))) ** 2 * int(
            self.input_size / (2 ** (self.stage + 3)) + 1) ** 2

        self.Conv = nn.Sequential(nn.Conv2d(in_channels=2 ** self.stage, out_channels=2 ** self.stage,
                                            kernel_size=3, stride=1, padding=1),
                                  nn.Softmax(-1),
                                  nn.InstanceNorm2d(num_features=int(self.num_features)))
        self.softmax = nn.Softmax(-1)
        self.fc_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout_ratio)

        )

    def forward(self, feats):
        k, v = self.trans_kv(feats).chunk(2, dim=-1)
        q = self.trans_q(feats)
        q, k, v = rearrange(q, 'b n (h d) -> b n h d', h=self.head), \
                  rearrange(k, 'b n (h d) -> b n h d', h=self.head), \
                  rearrange(v, 'b n (h d) -> b n h d', h=self.head)
        alpha = einsum('b i h d, b j h d -> b h i j', q, k) * self.scale
        att = self.Conv(alpha)
        att = self.softmax(alpha)
        out = einsum('b h i j, b j h d -> b h i d', att, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.fc_out(out)
        return out
