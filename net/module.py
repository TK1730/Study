import torch 
import torch.nn as nn
import torch.nn.functional as F

def Conv2d1x1(in_channel, out_channel, bias=True):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, dilation=1, bias=bias)


# 因果的な畳み込み
class DilatedCausalConv2dBlock(nn.Module):
    def __init__(self, 
                 in_channels, # 入力チャンネル
                 out_channels, # 出力チャンネル
                 kernel_size, # カーネルサイズ
                 stride, # ストライド
                 dilation, # ディレイション
                 **kwargs):
        super().__init__()
        # 因果性を持たせるためのパディング
        self.padding_w = (kernel_size[0] - 1) * dilation

        # 畳み込み
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding=(self.padding_w, 0), 
            **kwargs)
        
        # 活性化関数

        # 初期化
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # 2次元畳み込み
        x = self.conv(x)

        # 因果性を担保するために、順方向にシフトする
        if self.padding_w > 0:
            x = x[:, :, :-self.padding_w, :]

        return x
    

class CausalConvTranspose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, **kwargs):
        super().__init__()
        # 因果性を持たせる畳み込み
        self.padding_w = (kernel_size[0] - 1)
        # 畳み込み
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size, stride, 
            padding=padding,
            output_padding=output_padding,
            **kwargs)
        
        # 活性化関数
        self.mish = nn.Mish()

        # 初期化
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # 2次元畳み込み
        x = self.conv(x)
        # # 因果性を担保するために、順方向にシフトする
        # if self.padding_w > 0:
        #     x = x[:, :, :-self.padding_w, :]
        
        # 活性化関数
        x = self.mish(x)
        
        ln = nn.LayerNorm(x.shape[1:], elementwise_affine=False)
        x = ln(x)

        return x


class Resblock2d(nn.Module):
    def __init__(
            self,
            residual_chanels, # 残差結合のチャネル数
            gate_channels, # ゲートのチャネル数
            kernel_size, # カーネルサイズ
            stride,
            dilation,
            *args,
            **kwargs,
    ):
        super().__init__()
        # 因果性を持たせるためのパディング
        self.padding_w = (kernel_size[0] - 1) * dilation
        self.padding_h = (kernel_size[1] - 1) // 2

        # 因果畳み込み
        self.conv = nn.Conv2d(
            residual_chanels,
            gate_channels,
            kernel_size,
            padding=[self.padding_w, self.padding_h],
            stride=stride,
            dilation=dilation,
            *args,
            **kwargs
        )

        # 残差接続の畳み込み
        gate_channels = gate_channels // 2
        self.conv1x1_out = Conv2d1x1(gate_channels, residual_chanels)
        
        # 活性化関数
        self.mish = nn.Mish()
        
        # 初期化
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.xavier_normal_(self.conv1x1_out.weight)
        nn.init.zeros_(self.conv1x1_out.bias)
        
    def forward(self, x):
        # 残差接続用に入力を保持
        residual = x
        # 2次元畳み込み
        x = self.conv(x)
        # 因果性を保持するために出力をシフトする
        x = x[:, :, :-self.padding_w]
        # チャネル方向に分割
        a, b = torch.chunk(x, 2, dim=1)
        # ゲート付き活性化関数
        x = torch.tanh(a) * torch.sigmoid(b)

        # 残差接続用の要素和を行う前に次元数を合わせる
        x = self.conv1x1_out(x)

        x = x + residual

        ln = nn.LayerNorm(x.shape[1:], elementwise_affine=False)
        x = ln(x)

        return x

if __name__ == '__main__':
    resblock = Resblock2d(128, 256, (5, 5), stride=(1, 1),dilation=1)

    x = torch.randn(1, 128, 128, 80)
    x = resblock(x)
    print(x.shape)