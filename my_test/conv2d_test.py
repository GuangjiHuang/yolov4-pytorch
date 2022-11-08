import torch
from torch.nn.functional import conv2d
def my_conv2d(img, kernal, stride=1, padding=0):
    assert stride >= 1, "the stride must ge 1"
    i_h, i_w = img.shape[2:]
    k_h, k_w = kernal.shape[2:]
    assert i_h>=k_h and i_w>=k_w, "the image size must ge the kernal"
    o_h = (i_h - k_h + 2*padding) // stride + 1
    o_w = (i_w - k_w + 2*padding) // stride + 1
    # padding
    if padding != 0:
        n_img = torch.zeros(img.shape[0], img.shape[1], i_h+padding*2, i_w+padding*2)
        n_img[:, :, padding:-padding, padding:-padding] = img
        img = n_img
    # creat the new tensor
    out = torch.zeros((img.shape[0], img.shape[1], o_h, o_w))
    for i in range(o_h):
        for j in range(o_w):
            # strid * i, and the strid * j
                for k_i in range(k_h):
                    for k_j in range(k_w):
                        out[:, :, i, j] += kernal[:, :, k_i, k_j] * img[:, :, i*stride+k_i, j*stride+k_j]
    return out

torch.random.manual_seed(0)
img = torch.randn(1, 1, 6, 6)
kernal = torch.randn(1, 1, 3, 3)

out = conv2d(img, kernal, stride=2, padding=1)
my_out = my_conv2d(img, kernal, 2, 1)
print("pytorch computing: ")
print(out)
print("my function computing: ")
print(my_out)