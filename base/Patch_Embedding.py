from torch import nn
from einops.layers.torch import Rearrange


class Patch_Embed(nn.Module):
    """docstring for Patch_Embed"""

    def __init__(self, patch_size, input_channels, input_shape, dim):
        super(Patch_Embed, self).__init__()
        assert 'Input\'s shape must be 4 or 5', (input_shape == 4) or (input_shape == 5)
        if input_shape == 4:
            self.Input2Patch = Rearrange('b c (p1 w) (p2 h) -> b p1 p2 (c w h)', w=patch_size, h=patch_size)
        else:
            self.Input2Patch = Rearrange('b n c (p1 w) (p2 h) -> b n p1 p2 (c w h)', w=patch_size, h=patch_size)
        self.Embed = nn.Linear(input_channels * patch_size ** 2, dim)

    def forward(self, feats):
        feats = self.Input2Patch(feats)
        feats = self.Embed(feats)
        return feats
