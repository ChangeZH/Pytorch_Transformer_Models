from torch import nn


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
