import torch




bbox1 = torch.Tensor(20, 4)
bbox2 = torch.Tensor(20, 4)

inter_max_xy = torch.min(bbox1[:, 2:], bbox2[:, 2:])

inter_l = torch.max(bbox1, bbox2)