import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import numpy as np
import onnxruntime


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class MyConv2d(nn.Module):
    def __init__(self, in_c, out_c, k_sz, stride=1, padding=0, bias=False):
        super(MyConv2d, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, k_sz, stride, padding, bias=bias),
                                  nn.BatchNorm2d(out_c),
                                  Mish())
    def forward(self, x):
        out = self.conv(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, is_first=False):
        super(ResidualBlock, self).__init__()
        if is_first:
            self.conv_part = nn.Sequential(MyConv2d(in_c, out_c//2, 1, 1, 0),
                                       MyConv2d(out_c//2, out_c, 3, 1, 1))
        else:
            self.conv_part = nn.Sequential(MyConv2d(in_c, out_c, 1, 1, 0),
                                           MyConv2d(out_c, out_c, 3, 1, 1))
    def forward(self, x):
        out = self.conv_part(x)
        out = out + x
        return out

class CSPNet(nn.Module):
    def __init__(self, in_c, out_c, res_num, is_first=False):
        super(CSPNet, self).__init__()
        self.front = MyConv2d(in_c, out_c, 3, 2, 1)
        if is_first:
            self.left = nn.Sequential(MyConv2d(out_c, out_c, 1, 1, 0),
                                      ResidualBlock(out_c, out_c, is_first=True),
                                      MyConv2d(out_c, out_c, 1, 1, 0))
            self.right = MyConv2d(out_c, out_c, 1, 1, 0)
            self.back = MyConv2d(2*out_c, out_c, 1, 1, 0)
        else:
            res_list = []
            for i in range(res_num):
                res_list.append(ResidualBlock(out_c//2, out_c//2))
            self.left = nn.Sequential(MyConv2d(out_c, out_c//2, 1, 1, 0),
                                      nn.Sequential(*res_list),
                                      MyConv2d(out_c//2, out_c//2, 1, 1, 0))
            self.right = MyConv2d(out_c, out_c//2, 1, 1, 0)
            self.back = MyConv2d(out_c, out_c, 1, 1, 0)

    def forward(self, x):
        out = self.front(x)
        l = self.left(out)
        r = self.right(out)
        out = torch.cat([l, r], dim=1) # [batch_size, channels, w, h
        out = self.back(out)
        return out


class CspDarknet(nn.Module):
    def __init__(self):
        super(CspDarknet, self).__init__()
        self.conv1 = MyConv2d(3, 32, 3, 1, 1)
        self.csp_1 = CSPNet(32, 64, 1, is_first=True)
        self.csp_2 = CSPNet(64, 128, 2)
        self.csp_3 = CSPNet(128, 256, 4)
        self.csp_4 = CSPNet(256, 512, 8)
        self.csp_5 = CSPNet(512, 1024, 8)

    def forward(self, x):
        out = self.conv1(x)
        out_1 = self.csp_1(out)
        out_2 = self.csp_2(out_1)
        out_3 = self.csp_3(out_2)
        out_4 = self.csp_4(out_3)
        out_5 = self.csp_5(out_4)
        return out_3, out_4, out_5

class SPPNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(SPPNet, self).__init__()
        self.front_3_conv = nn.Sequential(MyConv2d(in_c, out_c, 1, 1, 0),
                                          MyConv2d(out_c, 2*out_c, 3, 1, 1),
                                          MyConv2d(2*out_c, out_c, 1, 1, 0))
        self.back_3_conv = nn.Sequential(MyConv2d(4*out_c, out_c, 1, 1, 0),
                                         MyConv2d(out_c, 2*out_c, 3, 1, 1),
                                         MyConv2d(2*out_c, out_c, 1,1, 0))
        self.max_p1 = nn.MaxPool2d(5, 1, 2)
        self.max_p2 = nn.MaxPool2d(9, 1, 4)
        self.max_p3 = nn.MaxPool2d(13, 1, 6)
    def forward(self, x):
        out = self.front_3_conv(x)
        maxp_1 = self.max_p1(out)
        maxp_2 = self.max_p2(out)
        maxp_3 = self.max_p3(out)
        out = torch.cat([maxp_1, maxp_2, maxp_3, out], dim=1)
        out = self.back_3_conv(out)
        return out

class ConvUps(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = MyConv2d(in_c, out_c, 1, 1, 0)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(self, x):
        out = self.conv(x)
        out = self.upsample(out)
        return out

class MakeFiveConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv5 = nn.Sequential(MyConv2d(in_c, out_c, 1, 1, 0),
                                   MyConv2d(out_c, in_c, 3, 1, 1),
                                   MyConv2d(in_c, out_c, 1, 1, 0),
                                   MyConv2d(out_c, in_c, 3, 1, 1),
                                   MyConv2d(in_c, out_c, 1, 1, 0))
    def forward(self, x):
        out = self.conv5(x)
        return out


class YoloHead(nn.Module):
    pass

class YOLO(nn.Module):
    def __init__(self, cl_num):
        super().__init__()
        self.backbone = CspDarknet()
        self.sppnet = SPPNet(1024, 512)
        # the PA NET pre
        self.conv1 = MyConv2d(256, 128, 1, 1, 0)
        self.conv2 = MyConv2d(512, 256, 1, 1, 0)
        # the pA NET: upsample
        self.conv_ups_1 = ConvUps(512, 256)
        self.five_conv_1 = MakeFiveConv(512, 256)
        self.conv_ups_2 = ConvUps(256, 128)
        self.five_conv_2 = MakeFiveConv(256, 128)
        # the PA NET: downsample
        self.downsample_1 = MyConv2d(128, 256, 3, 2, 1)
        self.five_conv_3 = MakeFiveConv(512, 256)
        self.downsample_2 = MyConv2d(256, 512, 3, 2, 1)
        self.five_conv_4 = MakeFiveConv(1024, 512)
        # the YOLO head
        self.head_1 = MyConv2d(128, 3*(4+1+cl_num), 1, 1, 0)
        self.head_2 = MyConv2d(256, 3*(4+1+cl_num), 1, 1, 0)
        self.head_3 = MyConv2d(512, 3*(4+1+cl_num), 1, 1, 0)

        # init the parameter
        self.initParameters()

    def forward(self, x):
        out1, out2, out3 = self.backbone(x)
        out1 = self.conv1(out1)
        out2 = self.conv2(out2)
        out3 = self.sppnet(out3)
        # the PA NET
        out2_1 = torch.cat([self.conv_ups_1(out3), out2], dim=1)
        out2_1 = self.five_conv_1(out2_1)
        out1_1 = torch.cat([self.conv_ups_2(out2_1), out1], dim=1)
        out1_1 = self.five_conv_2(out1_1)
        pre_1 = self.head_1(out1_1) # the first prediction
        #
        out2_2 = torch.cat([self.downsample_1(out1_1), out2_1], dim=1)
        out2_2 = self.five_conv_3(out2_2)
        pre_2 = self.head_2(out2_2)
        #
        out3_1 = torch.cat([self.downsample_2(out2_2), out3], dim=1)
        out3_1 = self.five_conv_4(out3_1)
        pre_3 = self.head_3(out3_1)
        return pre_1, pre_2, pre_3

    def initParameters(self):
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":

    img = torch.randn(1, 3, 416, 416).cuda()

    # load the model
    # yolo = YOLO(80).cuda()
    yolo = torch.load("my_model.pth")
    out1, out2, out3 = yolo(img)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)

    # save the onnx
    if 1:
        with torch.no_grad():
            torch.onnx.export(
                yolo,
                img,
                "yolo.onnx",
                opset_version=11,
                input_names=["input"],
                output_names=["out1", "out2", "out3"]
            )
    # load the onnx and then check it
    if 1:
        onnx_model = onnx.load("yolo.onnx")
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print("Model incorrect")
        else:
            print("Model correct!")
    # use the onnx runtime inferencer
    ort_session = onnxruntime.InferenceSession("yolo.onnx")
    ort_inputs = {"input": np.array(img.cpu().detach())};
    ort_output = ort_session.run(['out1', 'out2', 'out3'], ort_inputs)

    #

    for out in ort_output:
        print(out.shape)




















