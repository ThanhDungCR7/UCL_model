import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _single, _pair, _triple

def _calculate_fan_in_and_fan_out(tensor): #input and output size for a tensor
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
    def sigma(self): #const sigma = log(1+e^rho)
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.mu.size()).cuda()
        return self.mu + self.sigma * epsilon


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, ratio=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))

        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_mu)
        gain = 1 ## Var[w] + sigma^2 = 2/fan_in  --(paper dropout...) 
        # -->  giúp đảm bảo rằng trọng số được khởi tạo có độ lớn phù hợp với kiến trúc mô hình và dữ liệu đầu vào, 
        # đồng thời cũng giúp tăng tính ổn định và tốc độ huấn luyện của mô hình



        total_var = 2/fan_in
        noise_var = total_var * ratio
        mu_var = total_var - noise_var   #noise thể hiện độ uncertainty của mu (var[w]) ban đầu

        noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std)-1)

        nn.init.uniform_(self.weight_mu, -bound, bound)    #khoi tao ban dau
        self.bias = nn.Parameter(torch.Tensor(out_features).uniform_(0,0))

        self.weight_rho = nn.Parameter(torch.Tensor(out_features,1).uniform_(rho_init,rho_init))
        self.weight = Gaussian(weight.mu, weight.rho)

    def forward(self, input, sample=False):
        if sample:
            weight = self.weight.sample()
            bias = self.bias
        else:
            weight = self.weight.mu
            bias = self.bias

        return F.linear(input, weight, bias)


class _BayesianConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                  stride, padding, dilation, transposed, output_padding, groups, bias, ratio):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
    
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size)) #do không xác định được shape của kernel nên để dưới dạng *, nó có thể tùy biến được
        _, fan_out = _calculate_fan_in_and_fan_out(self.weight_mu)
        total_var = 2 / fan_out
        noise_var = total_var * ratio
        mu_var = total_var - noise_var
        
        noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std)-1)
        
        nn.init.uniform_(self.weight_mu, -bound, bound)
        self.bias = nn.Parameter(torch.Tensor(out_channels).uniform_(0,0), requires_grad = bias)
        
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, 1, 1, 1).uniform_(rho_init,rho_init))
        
        self.weight = Gaussian(self.weight_mu, self.weight_rho)


class BayesianConv2D(_BayesianConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, ratio=0.25):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BayesianConv2D, self).__init__(in_channels, out_channels, kernel_size, 
                                             stride, padding, dilation, False, _pair(0), groups, bias, ratio)

    def forward(self, input, sample=False):
        if sample:
            weight = self.weight.sample()
        else:
            weight = self.weight.mu

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)