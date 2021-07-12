from torch import nn


class PA(nn.Module):
    """docstring for PA"""

    def __init__(self, dim):
        super(PA, self).__init__()
        self.dim = dim
        self.conv = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, stride=1, padding=1)

    def forward(self, feats):
        return self.conv(feats).sigmoid()
