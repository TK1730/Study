import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def Conv1d1x1(in_channels, out_channels, bias=True):
    return nn.Conv1d(
        in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )

def GateActivation(x):
    """ゲート付き活性化関数

    Args:
        x (torch.Tensor): モデルの出力値

    Returns:
        torch.Tensor: 活性化関数を通した値
    """
    a, b = torch.chunk(x, 2, dim=1)
    x = torch.tanh(a) * torch.sigmoid(b)
    return x

class WN(nn.Module):
    def __init__(self,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                p_dropout=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels,
                                       2*hidden_channels,
                                       kernel_size,
                                       dilation=dilation,
                                       padding=padding)
            in_layer = nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # 最後の一つは必要ない
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            
            res_skip_layer = torch.nn.Conv1d(hidden_channels,
                                             res_skip_channels,
                                             1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)
    
    def forward(self, x):
        output = torch.zeros_like(x)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            # 活性化関数
            acts = GateActivation(x_in)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = x + res_acts
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output
    
    def remove_weight_norm(self):
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)

class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 frame_length,
                 n_layers):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.frame_length = frame_length
        self.n_layers = n_layers
        
        self.pre = Conv1d1x1(in_channels, hidden_channels)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels * self.frame_length, 4096),
            nn.Mish(inplace=True),
            nn.Linear(4096, 97),
        )
    
    def forward(self, x):
        x = self.pre(x)
        x = self.enc(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x

    
