import torch,pdb
import torchvision
import torch.nn.modules
import torch.nn as nn

class vgg16bn(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(vgg16bn,self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)
        
    def forward(self,x):
        return self.model(x)


class QuantDequantLayer(nn.Module):
    def __init__(self, bitwidth=8):
        super().__init__()
        assert bitwidth in [8, 16], "Only int8 or int16 are supported"
        self.bitwidth = bitwidth

    def forward(self, x):
        if self.bitwidth == 8:
            qmin, qmax = -128, 127
            dtype = torch.int8
        elif self.bitwidth == 16:
            qmin, qmax = -32768, 32767
            dtype = torch.int16

        min_val = x.min()
        max_val = x.max()
        
        # Asymmetric scale and zero-point
        scale = (max_val - min_val) / float(qmax - qmin)
        zero_point = qmin - (min_val / scale)
        zero_point = int(round(zero_point))
        zero_point = max(qmin, min(qmax, zero_point))  # clamp helpful for instances where range is [2,10] and 0 isnt there


        # Quantize
        qx = torch.clamp((x / scale).round() + zero_point, qmin, qmax).to(dtype)

        # Dequantize
        x_dequant = (qx.to(torch.float32) - zero_point) * scale

        return x_dequant



# Fixed-point quantization module
class Quantize(nn.Module):
    def __init__(self, IWL=4, FWL=4):
        super().__init__()
        self.scale = 2 ** FWL
        self.max_val = (2 ** IWL) - (1 / self.scale)
        self.min_val = - (2 ** (IWL - 1))  # Change to 0 if unsigned
        self.IWL = IWL
        self.FWL = FWL

    def forward(self, x):
        x = torch.clamp(x, self.min_val, self.max_val)
        return torch.floor(x * self.scale + 0.5) / self.scale


class resnet(nn.Module):
    def __init__(self, layers, pretrained=False):
        super(resnet, self).__init__()
        model = torchvision.models.resnet18(weights=pretrained)

        # Add quantization module
        # self.quant = Quantize(4, 4)
        self.quant = QuantDequantLayer(bitwidth=8)

        self.conv1 = nn.Sequential(model.conv1, self.quant)
        self.bn1 = nn.Sequential(model.bn1, self.quant)
        self.relu = model.relu
        self.maxpool = model.maxpool

        # Quantize after each layer group
        self.layer1 = nn.Sequential(model.layer1, self.quant)
        self.layer2 = nn.Sequential(model.layer2, self.quant)
        self.layer3 = nn.Sequential(model.layer3, self.quant)
        self.layer4 = nn.Sequential(model.layer4, self.quant)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x2, x3, x4
