import cv2
import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange


class PA(nn.Module):
    """docstring for PA"""

    def __init__(self, dim):
        super(PA, self).__init__()
        self.dim = dim
        self.conv = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, stride=1, padding=1)

    def forward(self, feats):
        return self.conv(feats).sigmoid()


class Stem(nn.Module):
    """docstring for Stem"""

    def __init__(self, input_channel, dim):
        super(Stem, self).__init__()
        self.input_channel = input_channel
        self.dim = dim
        self.Stem_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=self.dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dim // 2, out_channels=self.dim // 2,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dim // 2, out_channels=self.dim, kernel_size=3, stride=2, padding=1))
        self.PA_layer = PA(dim=self.dim)

    def forward(self, feats):
        feats = self.Stem_layer(feats)
        return feats * self.PA_layer(feats)


class Patch_Embedding(nn.Module):
    """docstring for Patch_Embedding"""

    def __init__(self, dim, head):
        super(Patch_Embedding, self).__init__()

    def forward(self, feats):
        return feats


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
        print(q.shape, k.shape, v.shape, self.dim, int(self.input_size // 2 ** (self.stage + 2)), 2 ** self.stage)
        alpha = einsum('b i h d, b j h d -> b h i j', q, k) * self.scale
        print(alpha.shape, self.Conv)
        att = self.Conv(alpha)
        # att = self.softmax(alpha)
        out = einsum('b h i j, b j h d -> b h i d', att, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.fc_out(out)
        return out


class MLP(nn.Module):
    """docstring for MLP"""

    def __init__(self, dim, inter_dim, dropout_ratio):
        super(MLP, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, inter_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(inter_dim, dim, bias=False),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        return self.ff(x)


class ResT(nn.Module):
    """docstring for ResT"""

    def __init__(self,
                 num_classes=10,
                 input_size=224,
                 input_channel=3,
                 patch_size=1,
                 dim=64,
                 stage=4,
                 depth=[2, 2, 2, 2],
                 head=[1, 2, 4, 8],
                 dropout_ratio=0.1):
        super(ResT, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.input_channel = input_channel
        self.patch_size = patch_size
        self.dim = dim
        self.stage = stage
        self.depth = depth
        self.head = head
        self.dropout_ratio = dropout_ratio
        self.stem = nn.Sequential(Stem(input_channel=self.input_channel, dim=self.dim),
                                  Rearrange('b c w h -> b (w h) c'))
        self.ResT = nn.ModuleDict(
            {str(i): nn.ModuleList(
                [nn.Sequential(EMSA(input_size=input_size, stage=i, dim=self.dim * 2 ** i, head=self.head[i],
                                    dropout_ratio=self.dropout_ratio),
                               MLP(dim=self.dim * 2 ** i, inter_dim=4 * self.dim * 2 ** i,
                                   dropout_ratio=self.dropout_ratio))
                 for _ in range(self.depth[i])]) for i in range(self.stage)})

        self.PatchEmbed = nn.ModuleDict(
            {str(i): nn.Sequential(nn.Conv2d(in_channels=2 ** (i - 1) * self.dim, out_channels=2 ** i * self.dim,
                                             kernel_size=3, stride=2, padding=1),
                                   PA(dim=2 ** i * self.dim),
                                   Rearrange('b c w h -> b (w h) c'))
             for i in range(1, self.stage)})

        self.UnEmbed = nn.ModuleDict(
            {str(i): nn.Sequential(Rearrange('b (p1 p2) (c w h) -> b c (p1 w) (p2 h)',
                                             p1=self.input_size // (2 ** (i + 2) * self.patch_size),
                                             p2=self.input_size // (2 ** (i + 2) * self.patch_size),
                                             c=self.dim * 2 ** i, w=self.patch_size, h=self.patch_size))
             for i in range(self.stage)})
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.dim * 2 ** (self.stage - 1), self.num_classes)

    def forward(self, feats):
        self.batchsize = feats.shape[0]
        feats = self.stem(feats)
        for stage in range(self.stage):
            feats = self.PatchEmbed[str(stage)](feats) if str(stage) in self.PatchEmbed else feats
            for block in self.ResT[str(stage)]:
                feats = block(feats)
            feats = self.UnEmbed[str(stage)](feats)
        feats = self.GlobalAvgPool(feats)
        feats = self.classifier(feats.reshape(self.batchsize, -1))
        return feats
