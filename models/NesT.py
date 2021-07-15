import torch
from base import *
from torch import nn
from einops.layers.torch import Rearrange


class NesT_Block(nn.Module):
    """docstring for NesT_Block"""

    def __init__(self, dim, head, inter_dim, dropout_ratio):
        super(NesT_Block, self).__init__()
        self.att = MSA(dim, inter_dim, head, dropout_ratio)
        self.ffn = MLP(dim, inter_dim, dropout_ratio)
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
                 num_classes,
                 input_size,
                 input_channel,
                 patch_size,
                 depth,
                 dim,
                 block_depth,
                 head,
                 dropout_ratio,
                 **kwargs):
        super(NesT, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.input_channel = input_channel
        self.patch_size = patch_size
        self.depth = depth
        self.dim = dim
        self.block_depth = block_depth
        self.head = head
        self.dropout_ratio = dropout_ratio
        self.kwargs = kwargs

        self.block_num = {i: 4 ** (len(self.block_depth) - i - 1) for i in range(len(self.block_depth))}
        self.agg_index = [i for i in range(len(self.block_depth) - 1)]
        self.PatchEmbed = nn.Sequential(
            Rearrange('b c (p1 w) (p2 h) -> b (p1 p2) (c w h)', w=self.patch_size, h=self.patch_size),
            nn.Linear(self.patch_size ** 2 * self.input_channel, self.dim))
        self.pos_encodings = {nn.Parameter(torch.zeros(self.block_num[i], self.dim)) for i in self.block_num}
        self.Block = {str(i): Rearrange('b (k p) d -> b k p d', k=self.block_num[i]) for i in self.block_num}
        self.UnBlock = {str(i): nn.Sequential(Rearrange('b k p d -> b (k p) d'),
                                              Rearrange('b (w h) d -> b w h d',
                                                        w=self.input_size // (2 ** i * self.patch_size),
                                                        h=self.input_size // (2 ** i * self.patch_size)))
                        for i in self.block_num}
        self.Agg_Block = nn.ModuleDict({str(i): Rearrange('b c (p1 w) (p2 h) -> b (p1 p2) c w h',
                                                          p1=int(self.block_num[i + 1] ** .5),
                                                          p2=int(self.block_num[i + 1] ** .5))
                                        for i in range(len(self.block_depth) - 1)})
        self.Agg_UnBlock = nn.ModuleDict({str(i): Rearrange('b (p1 p2) c w h -> b c (p1 w) (p2 h)',
                                                            p1=int(self.block_num[i + 1] ** .5),
                                                            p2=int(self.block_num[i + 1] ** .5))
                                          for i in range(len(self.block_depth) - 1)})
        self.Agg_Block_Conv = nn.ModuleDict({str(i): nn.ModuleDict({str(j): nn.Conv2d(self.dim, self.dim, 3, 1, 1)
                                                                    for j in range(self.block_num[i + 1])})
                                             for i in range(len(self.block_depth) - 1)})
        self.Agg_Image_Conv = nn.ModuleDict({str(i): nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.dim),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            for i in range(len(self.block_depth) - 1)})

        self.T = nn.ModuleDict(
            {str(i): nn.ModuleDict(
                {str(j): nn.ModuleList(
                    [NesT_Block(dim=self.dim, head=self.head, inter_dim=self.dim, dropout_ratio=self.dropout_ratio)
                     for _ in range(self.depth)])
                    for j in range(self.block_num[i])}) for i in self.block_num})
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.dim, self.num_classes)

    def Aggregate(self, y, i):
        z = self.UnBlock[i](y).permute(0, 3, 1, 2)
        z = self.Agg_Block[i](z)
        z = {str(j): z[:, j] for j in range(z.shape[1])}
        for _ in z:
            z[_] = self.Agg_Block_Conv[i][_](z[_])
        z = torch.stack([z[j] for j in z], dim=1)
        z = self.Agg_UnBlock[i](z)
        z = self.Agg_Image_Conv[i](z).permute(0, 2, 3, 1).reshape(self.batchsize, -1, self.dim)
        z = self.Block[str(int(i) + 1)](z)
        return z

    def forward(self, feats):
        self.batchsize = feats.shape[0]
        ids = [str(i) for i in range(self.depth)]
        feats = self.PatchEmbed(feats)
        feats = self.Block['0'](feats)
        for i in ids:
            feats = {str(j): feats[:, j] for j in range(feats.shape[1])}
            for j in feats:
                for layer in self.T[i][j]:
                    feats[j] = layer(feats[j])
            feats = torch.stack([feats[j] for j in feats], dim=1)
            if ids.index(i) < len(ids) - 1:
                feats = self.Aggregate(feats, i)
        feats = self.GlobalAvgPool(feats.permute(0, 3, 1, 2))
        feats = self.classifier(feats.reshape(self.batchsize, -1))
        return feats


if __name__ == '__main__':
    model = NesT()
    inputs = torch.ones(3, 1, 32, 32)
    outputs = model(inputs)
    print(outputs.shape)
