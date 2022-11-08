import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as FN


data = np.arange(24).reshape((2, 4, -1))
data_re = data.reshape((4, 3, 2), order='F')

data_transpose = np.transpose(data, (1, 0, 2))
data_transpose = data_transpose.copy(order='A')

print("original data")
print(data.flags)

print("reshape original data")
print(data_re.flags)

print("transpose data")
print(data_transpose.flags)
#print(data.flatten())
#print(data_transpose.flatten())