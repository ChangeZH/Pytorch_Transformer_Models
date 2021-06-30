import cv2
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    """docstring for FeedForward"""

    def __init__(self, dim, inter_dim, dropout_ratio):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, inter_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(inter_dim, dim, bias=False),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        return self.ff(x)


class Self_Attention(nn.Module):
    """docstring for Self_Attention"""

    def __init__(self, dim, head, inter_dim, dropout_ratio, type):
        super(Self_Attention, self).__init__()
        self.dim = dim
        self.head = head
        self.type = type
        self.inter_dim = inter_dim
        self.scale = inter_dim ** -.5
        if self.type == 'en':
            self.fc_qkv = nn.Linear(dim, inter_dim * 3, bias=False)
        elif self.type == 'de':
            self.fc_qk = nn.Linear(dim, inter_dim * 2, bias=False)
            self.fc_v = nn.Linear(dim, inter_dim, bias=False)
        self.softmax = nn.Softmax(-1)
        self.fc_out = nn.Sequential(
            nn.Linear(inter_dim, dim, bias=False),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, features):
        if self.type == 'en':
            q, k, v = self.fc_qkv(features).chunk(3, dim=-1)
        elif self.type == 'de':
            q, k = self.fc_qk(features[0]).chunk(2, dim=-1)
            v = self.fc_v(features[1])
        q, k, v = rearrange(q, 'b n p (h d) -> b n h p d', h=self.head), \
                  rearrange(k, 'b n p (h d) -> b n h p d', h=self.head), \
                  rearrange(v, 'b n p (h d) -> b n h p d', h=self.head)
        alpha = einsum('b n h i d, b n h j d -> b n h i j', q, k) * self.scale
        att = self.softmax(alpha)
        out = einsum('b n h i j, b n h j d -> b n h i d', att, v)
        out = rearrange(out, 'b n h p d -> b n p (h d)')
        return self.fc_out(out)


class Encoder_Layer(nn.Module):
    """docstring for Encoder_Layer"""

    def __init__(self, dim=16, head=8, inter_dim=32, dropout_ratio=0.1):
        super(Encoder_Layer, self).__init__()
        self.att = Self_Attention(dim, head, inter_dim, dropout_ratio, 'en')
        self.ffn = FeedForward(dim, inter_dim, dropout_ratio)
        self.norm1 = nn.LayerNorm(inter_dim)
        self.norm2 = nn.LayerNorm(inter_dim)

    def forward(self, x):
        feat = self.att(x)
        y = self.norm1(feat) + x
        feat = self.ffn(y)
        feat = self.norm2(feat) + y
        return feat


class NesT(nn.Module):
    """docstring for NesT"""

    def __init__(self,
                 input_size=32,
                 patch_size=1,
                 depth=3,
                 dim=192,
                 block_depth=[4, 4, 4],
                 head=3,
                 dropout_ratio=0.1):
        super(NesT, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.depth = depth
        self.dim = dim
        self.block_depth = block_depth
        self.head = head
        self.dropout_ratio = dropout_ratio

        self.block_num = {i: 4 ** (len(self.block_depth) - i - 1) for i in range(len(self.block_depth))}
        self.PatchEmbed = nn.Sequential(
            Rearrange('b c (p1 w) (p2 h) -> b (p1 p2) (c w h)', w=self.patch_size, h=self.patch_size),
            nn.Linear(self.patch_size ** 2, self.dim))
        # self.pos_encodings =
        self.Block = {str(i): Rearrange('b (k p) d -> b k p d', k=self.block_num[i]) for i in self.block_num}

        self.T = nn.ModuleDict(
            {str(i): nn.ModuleDict(
                {str(j): nn.ModuleList([Encoder_Layer(dim=self.dim, head=self.head, inter_dim=self.dim)
                                        for _ in range(self.depth)])
                 for j in range(self.block_num[i])}) for i in self.block_num})

    def forward(self, feats):
        a = self.PatchEmbed(feats)
        for i in ['0','1','2']:
            print(self.Block[i](a).shape)
        return feats
