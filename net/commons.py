import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def Conv1d1x1(in_channels, out_channels, bias=True):
    return nn.Conv1d(
        in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )

def Conv2d1x1(in_channels, out_channels, bias=True):
    """_summary_
        チャンネル数変更用のカーネルサイズ1の畳み込み
    Args:
        in_channels (int): 入力のチャネル数
        out_channels (int): 出力のチャネル数
        bias (bool, optional): バイアス設定. Defaults to True.

    Returns:
        Conv2d : カーネルサイズ1x1の畳み込み層
    """
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=1, 
        padding=0, 
        dilation=1,
        bias=bias
    )

def GateActivation(x):
    a, b = torch.chunk(x, 2, dim=1)
    x = torch.tanh(a) * torch.sigmoid(b)
    return x

def GatedActivation2d(x):
    """2次元のゲート付き活性化関数

    Args:
        x (torch): 2DConv後のデータ

    Returns:
        (torch): ゲート付き活性化関数後のデータ
    """
    a, b = torch.chunk(x, 2, dim=1)
    x = torch.tanh(a) * torch.sigmoid(b)
    return x

def get_padding(kernel, dilation):
    return int((dilation * kernel - dilation) / 2)
