import torch
from base import *
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchMerging(nn.Module):
    """docstring for PatchMerging"""

    def __init__(self, kernal_size, in_dim, out_dim):
        super(PatchMerging, self).__init__()
        self.PatchMerge = nn.Sequential(Rearrange('b (p1 w) (p2 h) d -> b p1 p2 (w h d)',
                                                  w=kernal_size, h=kernal_size),
                                        nn.Linear(kernal_size ** 2 * in_dim, out_dim),
                                        nn.LayerNorm(out_dim))

    def forward(self, feats):
        return self.PatchMerge(feats)


class SwinT(nn.Module):
    """docstring for SwinT"""

    def __init__(self,
                 num_classes,
                 input_size,
                 input_channels,
                 patch_size,
                 dim,
                 depth,
                 head,
                 win_size,
                 dropout_ratio,
                 **kwargs):
        super(SwinT, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.head = head
        self.win_size = win_size
        self.dropout_ratio = dropout_ratio
        self.kwargs = kwargs

        self.PatchEmbedding = nn.Linear(self.input_channels, self.dim)
        self.PatchMerging = nn.ModuleDict({str(stage): PatchMerging(kernal_size=self.patch_size,
                                                                    in_dim=self.dim,
                                                                    out_dim=self.dim)
        if stage == 0 else PatchMerging(kernal_size=2,
                                        in_dim=self.dim * 2 ** (stage - 1),
                                        out_dim=self.dim * 2 ** stage)
                                           for stage in range(4)})

        self.num_Transformer_per_stage = {stage: int((self.input_size /
                                                      (self.win_size * self.patch_size * 2 ** stage)) ** 2)
                                          for stage in range(4)}
        self.Input2Window = Rearrange('b (p1 h) (p2 w) d -> b (p1 p2) (h w) d', h=self.win_size, w=self.win_size)
        self.Window2Input = nn.ModuleDict({str(stage): Rearrange('b (p1 p2) (h w) d -> b (p1 h) (p2 w) d ',
                                                                 p1=int(self.num_Transformer_per_stage[stage] ** .5),
                                                                 p2=int(self.num_Transformer_per_stage[stage] ** .5),
                                                                 w=self.win_size, h=self.win_size)
                                           for stage in range(4)})
        self.Transformer = nn.ModuleDict(
            {str(stage): nn.ModuleList(
                [nn.ModuleDict({str(i): MSA(dim=self.dim * 2 ** stage, inter_dim=self.dim * 2 ** stage,
                                            head=self.head[stage], dropout_ratio=self.dropout_ratio)
                                for i in range(self.num_Transformer_per_stage[stage])})
                 for j in range(self.depth[stage])])
                for stage in range(4)})

        self.Globe_AvgPool = nn.AdaptiveAvgPool2d(1)
        self.Classifer = nn.Linear(self.dim * 2 ** 3, self.num_classes)

    def get_feature_id(self, stage):
        feture_id = torch.Tensor([[[i] for i in range(self.num_Transformer_per_stage[stage])]])
        feture_id = rearrange(feture_id, 'b (w h) d -> b w h d',
                              w=int(self.num_Transformer_per_stage[stage] ** .5),
                              h=int(self.num_Transformer_per_stage[stage] ** .5))
        feture_id = torch.repeat_interleave(feture_id, self.win_size, dim=1)
        feture_id = torch.repeat_interleave(feture_id, self.win_size, dim=2)
        return feture_id

    def forward(self, feats):
        batch_size = feats.shape[0]
        feats = self.PatchEmbedding(feats.permute(0, 2, 3, 1))
        for stage in range(4):
            feats = self.PatchMerging[str(stage)](feats)
            feture_id = self.get_feature_id(stage=stage)
            feture_id = feture_id
            for layer in self.Transformer[str(stage)]:
                feats = self.Input2Window(feats)
                feture_id = self.Input2Window(feture_id)
                feats = {str(i): feats[:, i] for i in range(feats.shape[1])}
                feture_id = {str(i): feture_id[:, i] for i in range(feture_id.shape[1])}
                for i in range(self.num_Transformer_per_stage[stage]):
                    feats[str(i)] = layer[str(i)](feats[str(i)], mask=feture_id[str(i)])
                feats = torch.stack([feats[i] for i in feats], dim=1)
                feture_id = torch.stack([feture_id[i] for i in feture_id], dim=1)
                feats = self.Window2Input[str(stage)](feats)
                feture_id = self.Window2Input[str(stage)](feture_id)
                feats = torch.roll(feats, shifts=(self.win_size // 2, self.win_size // 2), dims=(1, 2))
                feture_id = torch.roll(feture_id, shifts=(self.win_size // 2, self.win_size // 2), dims=(1, 2))
        feats = self.Globe_AvgPool(feats.permute(0, 3, 1, 2)).reshape(batch_size, -1)
        feats = self.Classifer(feats)
        return feats


if __name__ == '__main__':
    model = SwinT()
    inputs = torch.ones(3, 1, 32, 32)
    outputs = model(inputs)
    print(outputs.shape)
