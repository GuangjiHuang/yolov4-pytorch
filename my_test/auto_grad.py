import torch, torchvision

data = torch.tensor([1,2], device="cpu")
print(data)

data.to(torch.device("cuda:0"))
print(data.device)
