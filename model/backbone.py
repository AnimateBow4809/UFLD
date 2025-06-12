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


class QuantDequantLayer(nn.Module):
    def __init__(self, bitwidth=8, do=True):
        super().__init__()
        assert 1 <= bitwidth <= 32, "bitwidth must be between 1 and 32"
        self.bitwidth = bitwidth
        self.do = do

        # Calculate qmin and qmax for signed integers
        self.qmin = -(2 ** (bitwidth - 1))
        self.qmax = (2 ** (bitwidth - 1)) - 1

        # Select dtype based on bitwidth
        if bitwidth <= 8:
            self.dtype = torch.int8
        elif bitwidth <= 16:
            self.dtype = torch.int16
        else:
            self.dtype = torch.int32

    def forward(self, x):
        if not self.do:
            return x

        min_val = x.min()
        max_val = x.max()

        # Avoid division by zero if x is constant
        if max_val == min_val:
            return x.clone()

        scale = (max_val - min_val) / float(self.qmax - self.qmin)
        zero_point = self.qmin - (min_val / scale)
        zero_point = int(round(zero_point.item()))
        zero_point = max(self.qmin, min(self.qmax, zero_point))  # clamp zero_point

        # Quantize
        qx = torch.clamp((x / scale).round() + zero_point, self.qmin, self.qmax).to(self.dtype)

        # Dequantize
        x_dequant = (qx.to(torch.float32) - zero_point) * scale

        return x_dequant


class resnet(nn.Module):
    def __init__(self, layers, pretrained=False,bitwidth=8):
        super().__init__()
        model = torchvision.models.resnet18(weights=pretrained)

        self.quant = QuantDequantLayer(bitwidth=bitwidth, do=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.quant(self.conv1(x))
        x = self.quant(self.bn1(x))
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.quant(self.layer1(x))
        x2 = self.quant(self.layer2(x))
        x3 = self.quant(self.layer3(x2))
        x4 = self.quant(self.layer4(x3))

        return x2, x3, x4
