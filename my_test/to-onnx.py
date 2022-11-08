import torch
import torch.nn
#import onnx

model = torch.load("../model_data/ep108.pth")
model.eval()

input_name = ["input"]
output_name = ["output"]

x = torch.randn(1, 3, 416, 416, requires_grad=True)

torch.onnx.export(model, x, "best.onnx", intput_names=input_name, output_names=output_name, verbose=True)