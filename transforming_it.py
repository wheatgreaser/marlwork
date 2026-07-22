import torch
import torch.nn as nn
import math
import numpy as np

m = nn.AdaptiveAvgPool2d((2,1))
input_val = torch.tensor([[[[0, -1], [-2, -3]]], [[[-1, 0], [1, 0]]]]).float()
output = m(input_val)
print(output)

