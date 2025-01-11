import sys
sys.path.append("./")
import torch.utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from net import commons, modules

class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels,
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        
        self.pre = commons.Conv1d1x1(in_channels, hidden_channels)
        self.enc = modules.VITS_WN(hidden_channels, kernel_size, dilation_rate, n_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.pre(x)
        x = self.enc(x)
        x = self.proj(x)

        return x
    
class PosteriorEncoderVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        
        self.pre = commons.Conv1d1x1(in_channels, hidden_channels)
        self.enc = modules.VITS_WN(hidden_channels, kernel_size, dilation_rate, n_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels)

    def forward(self, x):
        x = self.pre(x)
        x = self.enc(x)
        stats = self.proj(x)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs)

        return z, m, logs

class PosteriorEncoder1d(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        
        self.pre = commons.Conv1d1x1(in_channels, hidden_channels)
        self.enc = modules.VITS_WN_Mish(hidden_channels, kernel_size, dilation_rate, n_layers)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels * 22, 4096),
            nn.Mish(inplace=True),
            nn.Linear(4096, 97),
        )
    
    def forward(self, x):
        x = self.pre(x)
        x = self.enc(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x


class PosteriorEncoder2d(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            dws_kernel_size,
            res_kernel_size,
            dws_stride_size,
            dilation_rate
            ):
        super().__init__()

        self.pre = commons.Conv2d1x1(in_channels, hidden_channels)
        self.enc = modules.WN(
            hidden_channels,
            dws_kernel_size,
            res_kernel_size,
            dws_stride_size,
            dilation_rate
            )
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels * 22 * 8, 4096),
            nn.Mish(inplace=True),
            nn.Linear(4096, 97),
        )
    
    def forward(self, x):
        x = self.pre(x)
        x = self.enc(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x


class PosteriorEncoder2ds(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            ):
        super().__init__()

        self.pre = commons.Conv2d1x1(in_channels, hidden_channels)
        self.enc = modules.wavenet(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels, 97),
        )
    
    def forward(self, x):
        x = self.pre(x)
        x = self.enc(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x

class Encoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels,
            dws_kernel_size,
            resblock_kernel_size,
            stride,
            dilation_rate,
            c_layers,
            n_layers,
            LayerNorm,
            p_dropout,
        ):
        super().__init__()

        self.pre = commons.Conv2d1x1(in_channels, hidden_channels)
        self.enc = modules.WN(
            hidden_channels=hidden_channels,
            gate_channels=hidden_channels//2,
            dws_kernel_size=dws_kernel_size,
            resblock_kernel_size=resblock_kernel_size,
            stride=stride,
            dilation_rate=dilation_rate,
            c_layers=c_layers,
            n_layers=n_layers,
            LayerNorm=LayerNorm,
            p_dropout=p_dropout
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels * 80 * 44, 4096),
            nn.Mish(inplace=True),
            nn.Linear(4096, 97),
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.enc(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class Decoder1d(nn.Module):
    def __init__(self, 
                 initial_channel,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_kernels,
                 upsample_rates,
                 upsample_initial_channel,
                 ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = commons.Conv1d1x1(initial_channel, upsample_initial_channel)
        resblock = modules.ResBlock1_1d

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernels)):
            self.ups.append(nn.utils.weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                   k, u, padding=(k-u)//2)
            ))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(commons.init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.mish(x)
            x = self.ups[i](x)
            
            for j in range(self.num_kernels):
                xs = self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.mish(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x
    
    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            nn.utils.remove_weight_norm(l)
        for l in self.resblocks:
            nn.utils.remove_weight_norm(l)


class Decoder2d(nn.Module):
    def __init__(self,
                 initial_channel,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_kernels,
                 upsample_rates,
                 upsample_initial_channel,
                 LayerNorm,
                 ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = commons.Conv2d1x1(initial_channel, upsample_initial_channel)
        resblock = modules.ResBlock1_2d

        self.ups = nn.ModuleList()
        for i, (r, k) in enumerate(zip(upsample_rates, upsample_kernels)):
            self.ups.append(
                modules.UPSConv(
                    upsample_initial_channel//(2**i),
                    upsample_initial_channel//(2**(i+1)),
                    k,
                    r,
                    padding=((k[0]-1)//2, (k[1]-r[1])//2),
                    LayerNorm=LayerNorm
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(
                    resblock(ch, k, d)
                )
        # 最適化いるかも
        self.conv_post = nn.Conv2d(ch, 1, (7, 1), 1, padding=(3, 0), bias=False)
    
    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.mish(x, inplace=False)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.mish(x, inplace=False)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class VocalNet(nn.Module):
    """
    encoderとdecoderを併せ持つ
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 inter_channels,
                 encoder_dwn_kernel_size,
                 encoder_resblock_kernel_size,
                 encoder_stride_size,
                 encoder_dilation_rate,
                 encoder_resblock_layers,
                 encoder_LayerNorm,
                 encoder_p_dropout,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernels,
                 decoder_LayerNorm,
                 **kwargs,
    ):
        super().__init__()
        self.enc = Encoder(
            in_channels,
            inter_channels,
            hidden_channels,
            encoder_dwn_kernel_size,
            encoder_resblock_kernel_size,
            encoder_stride_size,
            encoder_dilation_rate,
            len(encoder_dwn_kernel_size),
            encoder_resblock_layers,
            encoder_LayerNorm,
            encoder_p_dropout,
            )
        self.dec = Decoder(
            initial_channel=inter_channels,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_kernels=upsample_kernels,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            LayerNorm=decoder_LayerNorm,
        )
        self.initialize_weights()

    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)

        return enc, dec

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)


if __name__ == '__main__':
    x = torch.randn((1, 512, 1))
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    upsample_rates = [8, 8, 2, 2]
    upsample_initial_channel = 512
    upsample_kernel_sizes = [16, 16, 4, 4]
    model = Decoder1d(
        initial_channel=512,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_kernels=upsample_kernel_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel,
    )
    x = model(x)
    print(x.shape)
