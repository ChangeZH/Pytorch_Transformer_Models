import cv2
import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from NesT import NesT
from ResT import ResT

if __name__ == '__main__':
    model = ResT()
    inputs = torch.ones(3, 3, 224, 224)
    outputs = model(inputs)
    print(outputs.shape)
