from base import *
from torch import nn
from einops.layers.torch import Rearrange


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
