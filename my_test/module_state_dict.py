from torch import nn
import torch
import random
import numpy as np
import torchvision
import torch.nn.functional as F

torch.manual_seed(0)

x = torch.randint(1, 3, (10, 1, 4, 4))
x = x.to(dtype=torch.float32, device=torch.device("cuda:0"))


conv = torch.nn.Conv2d(1, 3, 2)
conv.cuda(0)
# calculate the conv
y = conv(x)

print(x)
print(x.shape)

print(y)
print(y.shape)
