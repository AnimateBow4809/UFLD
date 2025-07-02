import torch
from torchvision import transforms

class FixedPointQuantize:
    def __init__(self, IWL: int, FWL: int):
        self.IWL = IWL
        self.FWL = FWL
        self.scale = 2 ** FWL
        self.min_val = - (2 ** (IWL - 1))
        self.max_val = (2 ** (IWL - 1)) - (1 / self.scale)

    def __call__(self, tensor: torch.Tensor):
        """
        Simulates fixed-point quantization on a tensor.
        Assumes input is a float tensor in range [0, 1]
        """
        if self.FWL<0 or self.IWL < 0:
            return tensor
        
        # Clamp to fixed-point range
        tensor = torch.clamp(tensor, self.min_val, self.max_val)

        # Conventonal rounding book page 33   q= 1/scale
        quantized = torch.floor(tensor * self.scale + 0.5) / self.scale  # rounding

        return quantized

# 1.11101 -> 1111.00  ->1.111 