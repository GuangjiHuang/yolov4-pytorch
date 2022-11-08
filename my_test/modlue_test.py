import torch
import torch.nn as nn


# make my own module
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(x)

def func(m):
    print(m.__class__.__name__)
    print("hello, the world!")

if __name__ == "__main__":
    t = torch.Tensor(1)
    print(t)
    print(t.shape)
