import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _single, _pair, _triple

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2 :
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    
    if dimensions == 2:    # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        if tensor.dim() > 2:
            receptive_field = tensor[0][0].numel()   # số phần tử nằm trong tensor[0][0]
            fan_in = num_input_fmaps * receptive_field
            fan_out = num_output_fmaps * receptive_field

    return fan_in, fan_out


    class Gaussian(object):
        def __init__(self, mu, rho):
            super().__init__()
            self.mu = mu.cuda()
            self.rho = rho.cuda()
            self.normal = torch.distributions.Normal(0,1)

        @property
        def sigma(self):
            return torch.log1p(torch.exp(self.rho))

        def sample(self):
            epsilon = self.normal.sample(self.mu.size()).cuda()
            return self.mu + self.sigma * epsilon

    
    class BayessianLinear(nn.Module):
        def __init__(self, in_features, out_features, ratio=0.5):
            super().__init__()
