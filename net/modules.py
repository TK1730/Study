import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import sys
sys.path.append('./')
from net import commons


class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class LayerNormd2d(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x):
        mean = x.mean([1, 2, 3], keepdim=True)
        std = x.std([1, 2, 3], keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class GatedDilatedCausalConv1d(nn.Module):
    """ゲート付き活性化関数を用いた1次元の因果的膨張畳み込み"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels,
                              out_channels*2,
                              kernel_size,
                              padding=self.padding,
                              dilation=dilation)
        # 重みの初期化
        self.conv = weight_norm(self.conv, name='weight')

    def forward(self, x: torch.Tensor):
        # 1次元畳み込み
        x = self.conv(x)
        # 因果性を担保するために、順方向にシフトする
        if self.padding > 0:
            x = x[:, :, -self.padding]
        # ゲート付き活性化関数
        x = commons.GateActivation(x)

        return x


class GatedDilatedCausalConv2d(nn.Module):
    """
    ゲート付き活性化関数を用いた2次元膨張畳み込み
    """
    def __init__(
            self,
            in_channels,  # 入力チャンネル
            out_channels,  # 出力チャンネル
            kernel_size,  # カーネルサイズ
            stride,  # ストライド
            dilation,  # ディレイション
            *args,
            **kwargs):
        super().__init__()
        
        # 因果性を持たせるためのパディング
        self.padding = [0, 0]
        self.padding[0] = (kernel_size[0] - 1) * dilation[0]
        self.conv = nn.Conv2d(
            in_channels, # 入力チャンネル
            out_channels*2, # ゲート付き活性化関数のために2倍する
            kernel_size, # カーネルサイズ
            stride, # ストライド
            self.padding, # パディング
            dilation=dilation, # ディレイション
            *args,
            **kwargs,
        )
        # 重みの初期化
        self.conv = weight_norm(self.conv, name='weight')
    
    def forward(self, x):
        # 2次元畳み込み
        x = self.conv(x)
        # 因果性を担保するために、順方向にシフトする
        if self.padding[0] > 0:
            x = x[:, :, :-self.padding[0], :]

        # ゲート付き活性化関数
        x = commons.GatedActivation2d(x)

        return x


class ResSkipBlock1d(nn.Module):
    def __init__(
            self,
            residual_channels,
            gate_channels,
            skip_out_channels,
            kernel_size,
            dilation,
            *args,
            **kwargs,
            ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        # 一次元因果的膨張畳み込み (dilation == 1のときは、通常の1次元畳み込み)
        self.conv = nn.Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

        # ゲート付き活性化関数のために、１次元畳み込みの出力は２分割される
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = commons.Conv1d1x1(gate_out_channels, residual_channels)
        self.conv1x1_skip = commons.Conv2d1x1(gate_out_channels, skip_out_channels)

        # 重みの初期化
        self.conv = weight_norm(self.conv, name='weight')
        self.conv1x1_out = weight_norm(self.conv1x1_out, name='weight')
        self.conv1x1_skip = weight_norm(self.conv1x1_skip, name='weight')

    def forward(self, x: torch.Tensor):
        # 残差接続用に入力を保持
        residual = x
        # 1次元畳み込み
        x = self.conv(x)
        # 因果性を保証するために、出力をシフトする
        x = x[:, :-self.padding]
        # ゲート付き活性化関数
        x = commons.GateActivation(x)
        # スキップ接続用の出力を計算
        s = self.conv1x1_skip(x)
        # 残差接続の要素和を行う前に、次元数を合わせる
        x = self.conv1x1_out(x)
        # 残差接続
        x = x + residual

        return x, s
    

class ResBlock2d(nn.Module):
    """
    ゲート付き活性化関数を用いた2次元膨張畳み込みのResSkipBlock
    """
    def __init__(
            self,
            hidden_channels,
            kernel_size,
            dilation,
            *args,
            **kwargs,
            ):
        super().__init__()
        # paddingの設定
        # 因果性を持たせるパディング
        padding_w = (kernel_size[0] - 1) * dilation[0]
        # 出力を揃えるパディング
        padding_h = int((kernel_size[1] - 1) // 2)
        # list形式にする
        self.padding = [padding_w, padding_h]

        # 2次元因果的膨張畳み込み
        self.conv = nn.Conv2d(
            hidden_channels,
            hidden_channels*2,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            *args,
            **kwargs,
        )

        # 残差接続
        self.res_layer = commons.Conv2d1x1(hidden_channels, hidden_channels)

        # 重みの初期化
        self.conv = weight_norm(self.conv, name='weight')
        self.res_layer = weight_norm(self.res_layer, name='weight')
    
    def forward(self, x):
        # 2次元畳み込み
        x = self.conv(x)
        # 因果性を保証するために、出力をシフトする
        if self.padding[0] > 0:
            x = x[:, :, :-self.padding[0], :]
        # ゲート付き活性化
        x = commons.GatedActivation2d(x)
        # 残差接続用に入力を保持
        residual = x
        # 残差接続用の要素和を行う前に次元数を合わせる
        x = self.res_layer(x)
        # 要素和
        x = x + residual

        return x


class ResSkipBlock2d(nn.Module):
    """
    ゲート付き活性化関数を用いた2次元膨張畳み込みのResSkipBlock
    """
    def __init__(
            self,
            hidden_channels,
            kernel_size,
            dilation,
            *args,
            **kwargs,
            ):
        super().__init__()
        self.hidden_channels = hidden_channels
        # paddingの設定
        # 因果性を持たせるパディング
        padding_w = (kernel_size[0] - 1) * dilation[0]
        # 出力を揃えるパディング
        padding_h = int((kernel_size[1] - 1) // 2)
        # list形式にする
        self.padding = [padding_w, padding_h]

        # 2次元因果的膨張畳み込み
        self.conv = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels*2,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            *args,
            **kwargs,
        )

        # 残差接続とスキップ接続
        self.res_skip_layer = commons.Conv2d1x1(self.hidden_channels, self.hidden_channels*2)

        # 重みの初期化
        self.conv = weight_norm(self.conv, name='weight')
        self.res_skip_layer = weight_norm(self.res_skip_layer, name='weight')
    
    def forward(self, x):
        # 2次元畳み込み
        x = self.conv(x)
        # 因果性を保証するために、出力をシフトする
        if self.padding[0] > 0:
            x = x[:, :, :-self.padding[0], :]
        # ゲート付き活性化
        x = commons.GatedActivation2d(x)
        # 残差接続用に入力を保持
        residual = x
        # 残差接続とスキップ接続
        res_skip_acts = self.res_skip_layer(x)
        x = res_skip_acts[:, :self.hidden_channels, :, :]
        s = res_skip_acts[:, self.hidden_channels:, :, :]
        # 要素和
        x = x + residual

        return x, s

 
class WaveNet(nn.Module):
    def __init__(self,
                 out_channels=256,
                 layers=30,
                 stacks=3,
                 residual_channels=64,
                 gate_channels=128,
                 skip_out_channels=64,
                 kernel_size=2,
                 ):
        super().__init__()
        self.out_channels = out_channels
        
        self.first_conv = commons.Conv1d1x1(out_channels, residual_channels)

        # メインとなる畳み込み層
        self.main_conv_layers = nn.ModuleList()
        layers_per_stack = layers // stacks


class VITS_WN(nn.Module):
    def __init__(self,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 layernorm,
                p_dropout=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.layernorm = layernorm
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        if layernorm:
            self.layernorms = torch.nn.ModuleList()

        self.drop = nn.Dropout(p_dropout)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels,
                                       2*hidden_channels,
                                       kernel_size,
                                       dilation=dilation,
                                       padding=padding)
            in_layer = weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)
            if self.layernorm:
                in_layer = LayerNorm1d(hidden_channels*2)
                self.layernorms.append(in_layer)

            # 最後の一つは必要ない
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            
            res_skip_layer = torch.nn.Conv1d(hidden_channels,
                                             res_skip_channels,
                                             1)
            res_skip_layer = weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)
    
    def forward(self, x):
        output = torch.zeros_like(x)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if self.layernorm:
                x_in = self.layernorms[i](x_in)
            
            # 活性化関数
            acts = commons.GateActivation(x_in)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = x + res_acts
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output


class VITS_WN_Mish(nn.Module):
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
        self.mish = nn.Mish()

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels,
                                       hidden_channels,
                                       kernel_size,
                                       dilation=dilation,
                                       padding=padding)
            in_layer = weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # 最後の一つは必要ない
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            
            res_skip_layer = torch.nn.Conv1d(hidden_channels,
                                             res_skip_channels,
                                             1)
            res_skip_layer = weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)
    
    def forward(self, x):
        output = torch.zeros_like(x)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            # 活性化関数
            acts = self.mish(x_in)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = x + res_acts
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output


class WN(nn.Module):
    """
    2次元wavenet
    """
    def __init__(
            self,
            hidden_channels,
            dws_kernel_size,
            res_kernel_size,
            dws_stride_size,
            dilation_rate,
            p_dropout=0
            ):
        super().__init__()
        # 層の数
        self.c_layers = len(dws_kernel_size)
        self.r_layers = len(res_kernel_size)

        # メインとなる畳み込み層
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        for i in range(self.c_layers):
            # 畳み込み層
            gdcc2d = GatedDilatedCausalConv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=dws_kernel_size[i],
                stride=dws_stride_size[i],
                dilation=(1, 1),
            )
            self.in_layers.append(gdcc2d)

        # 残差接続とスキップ接続層
        for j in range(self.r_layers):
            dilation = dilation_rate ** j
            in_layer = ResSkipBlock2d(
                hidden_channels=hidden_channels,
                kernel_size=res_kernel_size[j],
                dilation=(dilation, 1)
            )
            self.res_skip_layers.append(in_layer)

        # 最後の層にスキップ接続必要なし
        self.conv_layer = ResBlock2d(
            hidden_channels=hidden_channels,
            kernel_size=res_kernel_size[j],
            dilation=(dilation, 1)
        )

    def forward(self, x):
        # 畳み込み層の処理
        for i in range(self.c_layers):
            # 周波数方向の畳み込み
            x = self.in_layers[i](x)
            # スキップ接続の出力を加算して保持
            skips = torch.zeros_like(x)
            for j in range(self.r_layers):
                # 最後の層だけスキップ接続の出力いらない
                if i < self.c_layers - 1 and j < self.r_layers-1:
                    x, s = self.res_skip_layers[j](x)
                    skips += s
                else:
                    x = self.conv_layer(x)
            
            x += skips

        return x


class wavenet(nn.Module):
    def __init__(
            self,
            hidden_channels,
            kernel_size,
            dilation_rate,
            p_dropout=0
            ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()

        for i in range(len(self.kernel_size)):
            dilation = dilation_rate ** i
            # 残差接続とスキップ接続層
            if i < len(self.kernel_size) - 1:
                in_layer = ResSkipBlock2d(
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size[i],
                    dilation=(dilation, 1)
                )
            else:
                in_layer = ResBlock2d(
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size[i],
                    dilation=(dilation, 1)
                )
            self.in_layers.append(in_layer)
    
    def forward(self, x):
        output = torch.zeros_like(x)
        for i in range(len(self.kernel_size)):
            if i < len(self.kernel_size) - 1:
                x, skip = self.in_layers[i](x)
                output += skip
            else:
                x = self.in_layers[i](x)
                output += x
        return x

    
# decoder用
class UPSConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            LayerNorm,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel,
            stride,
            padding
            )
        self.LayerNorm = LayerNorm
        self.layernorm = LayerNormd2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        if self.LayerNorm:
            x = self.layernorm(x)
        return x


class ResBlock1_1d(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=commons.get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=commons.get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=commons.get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(commons.init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=commons.get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=commons.get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=commons.get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(commons.init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.mish(x)
            xt = c1(xt)
            xt = F.mish(x)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            nn.utils.remove_weight_norm(l)
        for l in self.convs2:
            nn.utils.remove_weight_norm(l)


class ResBlock1_2d(nn.Module):
    def __init__(
            self,
            channels,
            kernel_size,
            dilation,
            ):
        super().__init__()
        self.channels = channels
        self.ln = LayerNormd2d(self.channels)

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.convs1 = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size[0], 1, dilation=dilation[0],
                               padding=(commons.get_padding(kernel_size[0], dilation[0][0]),
                                        commons.get_padding(kernel_size[0], dilation[0][1]))),
            
            nn.Conv2d(channels, channels, kernel_size[0], 1, dilation=dilation[1],
                               padding=(commons.get_padding(kernel_size[0], dilation[1][0]),
                                        commons.get_padding(kernel_size[0], dilation[1][1]))),
            
            nn.Conv2d(channels, channels, kernel_size[0], 1, dilation=dilation[2],
                               padding=(commons.get_padding(kernel_size[0], dilation[2][0]),
                                        commons.get_padding(kernel_size[0], dilation[2][1])))
        ])

        self.convs2 = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size[0], 1, dilation=1,
                               padding=(commons.get_padding(kernel_size[0], 1),
                                        commons.get_padding(kernel_size[0], 1))),
            
            nn.Conv2d(channels, channels, kernel_size[0], 1, dilation=1,
                               padding=(commons.get_padding(kernel_size[0], 1),
                                        commons.get_padding(kernel_size[0], 1))),

            nn.Conv2d(channels, channels, kernel_size[0], 1, dilation=1,
                               padding=(commons.get_padding(kernel_size[0], 1),
                                        commons.get_padding(kernel_size[0], 1))),
        ])

        
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.mish(x, inplace=False)
            xt = c1(xt)
            xt = self.ln(x)
            xt = F.mish(xt, inplace=False)
            xt = c2(xt)
            xt = self.ln(x)
            x = xt + x
        return x
    
if __name__ == '__main__':
    import torch
    import sys
    sys.path.append('./')
    wn = WN(256, ((3, 5), (3, 5)), ((5, 5), (5, 5), (5, 5)), ((1, 3), (1, 3)), 2)
    x = torch.randn(1, 256, 22, 80)
    x = wn(x)
    print(x.shape)
