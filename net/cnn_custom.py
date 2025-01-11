import torch
import torch.nn as nn
import torch.nn.functional as F
import net.module as module

# import modules

class Voice_encoder(nn.Module):
    def __init__(self, n_layers):
        super(Voice_encoder, self).__init__()
        self.n_layers


        # 畳み込み層
        self.conv1 = nn.Conv2d(self.in_channels, 3, kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(3, 27, kernel_size=self.kernel_size, padding=self.padding)
        # maxpooling層
        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)
        # 前結合層
        self.fc1 = nn.Linear(27 * 126 * 78, 1024)
        self.fc2 = nn.Linear(1024, 256, bias=False)
        # 活性化関数
        self.mish = nn.Mish()

    def forward(self, x):
        x = self.conv1(x)
        x = self.mish(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.mish(x)
        x = x.view(-1, 27 * 126 * 78)
        x = self.fc1(x)

        x = self.fc2(x)
        return x

class Voice_encoder_ver2(nn.Module):
    def __init__(self, in_channels=1, kernel_size=3, stride=3, padding=0, fc_size=1024, n_output=256, dropout=0.2):
        super(Voice_encoder_ver2, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 畳み込み層
        self.conv1 = nn.Conv2d(1, 64, kernel_size=self.kernel_size, padding=self.padding, stride=1)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=self.kernel_size, padding=self.padding, stride=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=self.kernel_size, padding=self.padding, stride=1)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=self.kernel_size, padding=self.padding, stride=1)

        # maxpooling層
        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size)
        # 前結合層
        self.fc1 = nn.Linear(512 * 3 * 2, fc_size)
        self.fc2 = nn.Linear(1024, n_output, bias=False)
        # 活性化関数
        self.mish = nn.Mish()

        self.dropout = nn.Dropout(dropout)
        # 重み初期化
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.fc1.bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.mish(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.mish(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = self.mish(x)

        # x = self.conv4(x)
        # x = self.mish(x)
        # print(x.shape)
        x = x.view(-1, 512 * 3 * 2)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.mish(x)

        x = self.fc2(x)
        return x

class Voice_encoder_ver3(nn.Module):
    def __init__(self, conv1_unit=64, conv2_unit=64, conv3_unit=64, linear1_unit=1024, linear2_unit=256, activate_func="mish", dropout=0.2):
        super(Voice_encoder_ver3, self).__init__()
        self.conv1_unit = conv1_unit
        self.conv2_unit = conv2_unit
        self.conv3_unit = conv3_unit
        self.linear1_unit = linear1_unit
        self.linear2_unit = linear2_unit
        self.dropout = dropout

        # 畳み込み層
        self.conv1 = nn.Conv2d(1, self.conv1_unit, kernel_size=3, padding=0, stride=1)
        self.conv2 = nn.Conv2d(self.conv1_unit, self.conv2_unit, kernel_size=3, padding=0, stride=1)
        self.conv3 = nn.Conv2d(self.conv2_unit, self.conv3_unit, kernel_size=3, padding=0, stride=1)

        # maxpooling層
        # if pool == 'MAX':
        self.pool_1 = nn.MaxPool2d(kernel_size=3)
        self.pool_2 = nn.MaxPool2d(kernel_size=3)
        self.pool_3 = nn.MaxPool2d(kernel_size=3)
        # elif pool == 'BN':
        #     self.pool_1 = nn.BatchNorm2d(self.conv1_unit)
        #     self.pool_2 = nn.BatchNorm2d(self.conv2_unit)
        #     self.pool_3 = nn.BatchNorm2d(self.conv3_unit)

        # 前結合層
        self.fc1 = nn.Linear(self.conv3_unit * 3 * 2, self.linear1_unit)
        self.fc2 = nn.Linear(self.linear1_unit, self.linear2_unit, bias=False)
        # 活性化関数
        if activate_func == "mish":
            self.activate = nn.Mish()
        elif activate_func == "swish":
            self.activate = nn.SiLU()
        elif activate_func == "tanh":
            self.activate = nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        # 重み初期化
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool_1(x)
        x = self.activate(x)

        x = self.conv2(x)
        x = self.pool_2(x)
        x = self.activate(x)

        x = self.conv3(x)
        x = self.pool_3(x)
        x = self.activate(x)

        x = x.view(-1, self.conv3_unit * 3 * 2)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activate(x)

        x = self.fc2(x)
        return x
    
class Voice_encoder_freq_conv(nn.Module):
    """
    周波数方向に畳み込みを行い類似度を算出する
    """
    def __init__(self, in_channels=1, kernel_size=(1, 5), stride=(1, 2), padding=0, fc_size=1024, n_output=256, dropout=0.2):
        super(Voice_encoder_freq_conv, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 畳み込み層
        self.conv1 = nn.Conv2d(self.in_channels, 256, kernel_size=self.kernel_size, padding=self.padding, stride=stride)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=self.kernel_size, padding=self.padding, stride=stride)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=self.kernel_size, padding=self.padding, stride=stride)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=self.kernel_size, padding=self.padding, stride=stride)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=self.kernel_size, padding=self.padding, stride=stride)

        # BatNorm層
        self.bn_1 = nn.BatchNorm2d(256)
        self.bn_2 = nn.BatchNorm2d(256)
        self.bn_3 = nn.BatchNorm2d(256)
        self.bn_4 = nn.BatchNorm2d(256)
        # 前結合層
        self.fc1 = nn.Linear(256 * 128 * 2, fc_size)
        self.fc2 = nn.Linear(1024, n_output, bias=False)
        # 活性化関数
        self.mish = nn.Mish()

        self.dropout = nn.Dropout(dropout)
        # 重み初期化
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)
        nn.init.zeros_(self.fc1.bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_1(x)
        # print(x.shape)
        x = self.mish(x)

        x = self.conv2(x)
        x = self.bn_2(x)
        x = self.mish(x)

        x = self.conv3(x)
        x = self.bn_3(x)
        x = self.mish(x)

        x = self.conv4(x)
        x = self.bn_4(x)
        x = self.mish(x)

        x = x.view(-1, 256 * 128 * 2)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.mish(x)

        x = self.fc2(x)
        return x

class Voice_encoder_freq_conv_ver2(nn.Module):
    """
    Voice_encoder_freq_convに音声復元を追加する
    """
    def __init__(self, in_channels=1, kernel_size=(1, 5), stride=(1, 2), padding=0, fc_size=1024, n_output=256, dropout=0.2):
        super(Voice_encoder_freq_conv_ver2, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # encoder
        self.conv1 = nn.Conv2d(1, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)

        self.conv_bn_1 = nn.BatchNorm2d(512)
        self.conv_bn_2 = nn.BatchNorm2d(512)
        self.conv_bn_3 = nn.BatchNorm2d(512)
        self.conv_bn_4 = nn.BatchNorm2d(512)

        # decoder
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))
        self.deconv3 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 1))
        self.deconv4 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 1))
        self.deconv5 = nn.Conv2d(512, 1, kernel_size=(1, 5), padding=(0, 2))

        self.deconv_bn_1 = nn.BatchNorm2d(512)
        self.deconv_bn_2 = nn.BatchNorm2d(512)
        self.deconv_bn_3 = nn.BatchNorm2d(512)
        self.deconv_bn_4 = nn.BatchNorm2d(512)

        # 前結合層
        self.fc1 = nn.Linear(512 * 128 * 2, fc_size)
        self.fc2 = nn.Linear(1024, n_output, bias=False)
        # 活性化関数
        self.mish = nn.Mish()

        self.dropout = nn.Dropout(dropout)
        # 重み初期化
        # encoder
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)

        # decoder
        nn.init.kaiming_normal_(self.deconv1.weight)
        nn.init.kaiming_normal_(self.deconv2.weight)
        nn.init.kaiming_normal_(self.deconv3.weight)
        nn.init.kaiming_normal_(self.deconv4.weight)
        nn.init.kaiming_normal_(self.deconv5.weight)
        nn.init.zeros_(self.deconv1.bias)
        nn.init.zeros_(self.deconv2.bias)
        nn.init.zeros_(self.deconv3.bias)
        nn.init.zeros_(self.deconv4.bias)
        nn.init.zeros_(self.deconv5.bias)

        # clip
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)


    def forward(self, x):
        x = self.encoder(x)
        dec_x = self.decoder(x)
        cli_x = self.clip(x)

        return cli_x, dec_x
    
    def encoder(self, x):
        self.enc_1 = self.mish(self.conv_bn_1(self.conv1(x)))
        self.enc_2 = self.mish(self.conv_bn_2(self.conv2(self.enc_1)))
        self.enc_3 = self.mish(self.conv_bn_3(self.conv3(self.enc_2)))
        self.enc_4 = self.mish(self.conv_bn_4(self.conv4(self.enc_3)))

        return self.enc_4
    
    def decoder(self, x):
        x = self.mish(self.deconv_bn_1(self.deconv1(x)))

        x = self.mish(self.deconv_bn_2(self.deconv2(x)))

        x = self.mish(self.deconv_bn_3(self.deconv3(x)))

        x = self.mish(self.deconv_bn_4(self.deconv4(x)))
        x = self.deconv5(x)

        return x

    def clip(self, x):
        x = x.view(-1, 512 * 128 * 2)
        x = self.fc1(x)
        x = self.mish(x)

        x = self.fc2(x)
        return x


class Voice_encoder_freq_conv_ver3(nn.Module):
    # clipの原文だと全結合層は1層しかない
    def __init__(self, in_channels=1, kernel_size=(1, 5), stride=(1, 2), padding=0, fc_size=512, n_output=256, dropout=0.2):
        super(Voice_encoder_freq_conv_ver3, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 畳み込み層
        self.conv1 = nn.Conv2d(1, 512, kernel_size=(1, 5), padding=self.padding, stride=(1, 2))
        self.conv2 = nn.Conv2d(512, 512, kernel_size=(1, 5), padding=self.padding, stride=(1, 2))
        self.conv3 = nn.Conv2d(512, 512, kernel_size=(1, 5), padding=self.padding, stride=(1, 2))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1, 5), padding=self.padding, stride=(1, 2))

        # batchnorm層
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv2_bn = nn.BatchNorm2d(512)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4_bn = nn.BatchNorm2d(512)

        # 逆転置畳み込み
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))
        self.deconv3 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 1))
        self.deconv4 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 1))
        self.deconv5 = nn.Conv2d(512, 1, kernel_size=(1, 5), padding=(0, 2))

        # batchnorm層
        self.deconv1_bn = nn.BatchNorm2d(512)
        self.deconv2_bn = nn.BatchNorm2d(512)
        self.deconv3_bn = nn.BatchNorm2d(512)
        self.deconv4_bn = nn.BatchNorm2d(512)

        # 前結合層
        self.fc1 = nn.Linear(512 * 128 * 2, fc_size, bias=False)

        # 活性化関数
        self.mish = nn.Mish()

        # ドロップ率
        self.dropout = nn.Dropout(dropout)
        
        # 重み初期化
        # encoder
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)
        
        # decoder
        nn.init.kaiming_normal_(self.deconv1.weight)
        nn.init.kaiming_normal_(self.deconv2.weight)
        nn.init.kaiming_normal_(self.deconv3.weight)
        nn.init.kaiming_normal_(self.deconv4.weight)
        nn.init.zeros_(self.deconv1.bias)
        nn.init.zeros_(self.deconv2.bias)
        nn.init.zeros_(self.deconv3.bias)
        nn.init.zeros_(self.deconv4.bias)

    def forward(self, x):
        x = self.encoder(x)
        v = self.decoder(x)
        x = self.projection(x)
        return x, v
    
    def encoder(self, x):
        x = self.conv1_bn(self.conv1(x))
        x = self.mish(x)

        x = self.conv2_bn(self.conv2(x))
        x = self.mish(x)

        x = self.conv3_bn(self.conv3(x))
        x = self.mish(x)

        x = self.conv4_bn(self.conv4(x))
        x = self.mish(x)
        return x
    
    def decoder(self, x):
        x = self.deconv1_bn(self.deconv1(x))
        x = self.mish(x)

        x = self.deconv2_bn(self.deconv2(x))
        x = self.mish(x)

        x = self.deconv3_bn(self.deconv3(x))
        x = self.mish(x)

        x = self.deconv4_bn(self.deconv4(x))
        x = self.mish(x)

        x = self.deconv5(x)
        return x
    
    def projection(self, x):
        x = x.view(-1, 512 * 128 * 2)
        x = self.fc1(x)
        return x

class Voice_encoder_freq_conv_ver4(nn.Module):
    """
    batchnormだと人ごとの分布をとれないため話者エンコードがうまくいかないのではと指摘を受けたのでぬく
    """
    def __init__(self, in_channels=1, kernel_size=(1, 5), stride=(1, 2), padding=0, fc_size=1024, n_output=256, dropout=0.2):
        super(Voice_encoder_freq_conv_ver4, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # encoder
        self.conv1 = nn.Conv2d(1, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)

        # decoder
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))
        self.deconv3 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 1))
        self.deconv4 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 1))
        self.deconv5 = nn.Conv2d(512, 1, kernel_size=(1, 5), padding=(0, 2))

        # 前結合層
        self.fc1 = nn.Linear(512 * 128 * 2, fc_size)
        self.fc2 = nn.Linear(1024, n_output, bias=False)
        # 活性化関数
        self.mish = nn.Mish()

        self.dropout = nn.Dropout(dropout)
        # 重み初期化
        # encoder
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)

        # decoder
        nn.init.kaiming_normal_(self.deconv1.weight)
        nn.init.kaiming_normal_(self.deconv2.weight)
        nn.init.kaiming_normal_(self.deconv3.weight)
        nn.init.kaiming_normal_(self.deconv4.weight)
        nn.init.kaiming_normal_(self.deconv5.weight)
        nn.init.zeros_(self.deconv1.bias)
        nn.init.zeros_(self.deconv2.bias)
        nn.init.zeros_(self.deconv3.bias)
        nn.init.zeros_(self.deconv4.bias)
        nn.init.zeros_(self.deconv5.bias)

        # clip
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)


    def forward(self, x):
        x = self.encoder(x)
        dec_x = self.decoder(x)
        cli_x = self.clip(x)

        return cli_x, dec_x
    
    def encoder(self, x):
        self.enc_1 = self.mish(self.conv1(x))
        self.enc_2 = self.mish(self.conv2(self.enc_1))
        self.enc_3 = self.mish(self.conv3(self.enc_2))
        self.enc_4 = self.mish(self.conv4(self.enc_3))

        return self.enc_4
    
    def decoder(self, x):
        x = self.mish(self.deconv1(x))

        x = self.mish(self.deconv2(x))

        x = self.mish(self.deconv3(x))

        x = self.mish(self.deconv4(x))
        x = self.deconv5(x)

        return x

    def clip(self, x):
        x = x.view(-1, 512 * 128 * 2)
        x = self.fc1(x)
        x = self.mish(x)

        x = self.fc2(x)
        return x
    
class Voice_encoder_freq_conv_ver5(nn.Module):
    def __init__(self, in_channels=1, kernel_size=(1, 5), stride=(1, 2), padding=0, fc_size=1024, n_output=256, dropout=0.2):
        super(Voice_encoder_freq_conv_ver5, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # encoder
        self.conv1 = nn.Conv2d(1, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)

        self.conv_ln_1 = nn.LayerNorm([512, 128, 38])
        self.conv_ln_2 = nn.LayerNorm([512, 128, 17])
        self.conv_ln_3 = nn.LayerNorm([512, 128, 7])
        self.conv_ln_4 = nn.LayerNorm([512, 128, 2])

        # decoder
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))
        self.deconv3 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 1))
        self.deconv4 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 1))
        self.deconv5 = nn.Conv2d(512, 1, kernel_size=(1, 5), padding=(0, 2))

        self.deconv_ln_1 = nn.LayerNorm([512, 128, 7])
        self.deconv_ln_2 = nn.LayerNorm([512, 128, 17])
        self.deconv_ln_3 = nn.LayerNorm([512, 128, 38])
        self.deconv_ln_4 = nn.LayerNorm([512, 128, 80])

        # 前結合層
        self.fc1 = nn.Linear(512 * 128 * 2, fc_size)
        self.fc2 = nn.Linear(1024, n_output, bias=False)
        # 活性化関数
        self.mish = nn.Mish()

        self.dropout = nn.Dropout(dropout)
        # 重み初期化
        # encoder
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)

        # decoder
        nn.init.kaiming_normal_(self.deconv1.weight)
        nn.init.kaiming_normal_(self.deconv2.weight)
        nn.init.kaiming_normal_(self.deconv3.weight)
        nn.init.kaiming_normal_(self.deconv4.weight)
        nn.init.kaiming_normal_(self.deconv5.weight)
        nn.init.zeros_(self.deconv1.bias)
        nn.init.zeros_(self.deconv2.bias)
        nn.init.zeros_(self.deconv3.bias)
        nn.init.zeros_(self.deconv4.bias)
        nn.init.zeros_(self.deconv5.bias)

        # clip
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)


    def forward(self, x):
        x = self.encoder(x)
        dec_x = self.decoder(x)
        cli_x = self.clip(x)

        return x, cli_x, dec_x
    
    def encoder(self, x):
        self.enc_1 = self.mish(self.conv_ln_1(self.conv1(x)))
        self.enc_2 = self.mish(self.conv_ln_2(self.conv2(self.enc_1)))
        self.enc_3 = self.mish(self.conv_ln_3(self.conv3(self.enc_2)))
        self.enc_4 = self.mish(self.conv_ln_4(self.conv4(self.enc_3)))

        return self.enc_4
    
    def decoder(self, x):
        x = self.mish(self.deconv_ln_1(self.deconv1(x)))

        x = self.mish(self.deconv_ln_2(self.deconv2(x)))

        x = self.mish(self.deconv_ln_3(self.deconv3(x)))

        x = self.mish(self.deconv_ln_4(self.deconv4(x)))
        x = self.deconv5(x)

        return x

    def clip(self, x):
        x = x.view(-1, 512 * 128 * 2)
        x = self.fc1(x)
        x = self.mish(x)

        x = self.fc2(x)
        return x
    

class Voice_encoder_freq_conv_ver6(nn.Module):
    def __init__(self, out_channels=1, layers=4, stacks=3, gate_channels=64, residual_channels=64, kernel_size=(1, 5), aux_context_window=0):
        super(Voice_encoder_freq_conv_ver6, self).__init__()
        self.out_channels = out_channels
        self.aux_context_window = aux_context_window

        self.first_conv = nn.Conv2d(out_channels, residual_channels, kernel_size=(1, 1))

        # メインとなる畳み込み層
        self.main_conv_layers = nn.ModuleList()
        layers_per_stack = layers // stacks
        for layer in range(layers):
            dilation = 1
            conv = module.ResBlock(
                residual_channels,
                gate_channels,
                kernel_size,
            )
            self.main_conv_layers.append(conv)

    def forward(self, x):
        # 入力チャネルの次元から隠れ層の次元に変換
        x = self.first_conv(x)

        # メインの畳み込み層の処理
        for f in self.main_conv_layers:
            x = f(x)
        
        return x

class Voice_encoder_freq_conv_ver7(nn.Module):
    def __init__(self, in_channels=1, kernel_size=(1, 5), stride=(1, 2), padding=0, fc_size=1024, n_output=256, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # encoder
        self.conv1 = nn.Conv2d(1, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv4 = nn.Conv2d(512, 1, kernel_size=kernel_size, padding=self.padding, stride=stride)

        self.conv_ln_1 = nn.LayerNorm([512, 128, 38])
        self.conv_ln_2 = nn.LayerNorm([512, 128, 17])
        self.conv_ln_3 = nn.LayerNorm([512, 128, 7])
        self.conv_ln_4 = nn.LayerNorm([1, 128, 2])

        # decoder
        self.deconv1 = nn.ConvTranspose2d(1, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 0))
        self.deconv3 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 1))
        self.deconv4 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0), output_padding=(0, 1))
        self.deconv5 = nn.Conv2d(512, 1, kernel_size=(1, 5), padding=(0, 2))

        self.deconv_ln_1 = nn.LayerNorm([512, 128, 7])
        self.deconv_ln_2 = nn.LayerNorm([512, 128, 17])
        self.deconv_ln_3 = nn.LayerNorm([512, 128, 38])
        self.deconv_ln_4 = nn.LayerNorm([512, 128, 80])

        # 活性化関数
        self.mish = nn.Mish()

        # 重み初期化
        # encoder
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)

        # decoder
        nn.init.kaiming_normal_(self.deconv1.weight)
        nn.init.kaiming_normal_(self.deconv2.weight)
        nn.init.kaiming_normal_(self.deconv3.weight)
        nn.init.kaiming_normal_(self.deconv4.weight)
        nn.init.kaiming_normal_(self.deconv5.weight)
        nn.init.zeros_(self.deconv1.bias)
        nn.init.zeros_(self.deconv2.bias)
        nn.init.zeros_(self.deconv3.bias)
        nn.init.zeros_(self.deconv4.bias)
        nn.init.zeros_(self.deconv5.bias)


    def forward(self, x):
        x = self.encoder(x)
        dec_x = self.decoder(x)

        return x, dec_x
    
    def encoder(self, x):
        self.enc_1 = self.mish(self.conv_ln_1(self.conv1(x)))
        self.enc_2 = self.mish(self.conv_ln_2(self.conv2(self.enc_1)))
        self.enc_3 = self.mish(self.conv_ln_3(self.conv3(self.enc_2)))
        self.enc_4 = self.mish(self.conv_ln_4(self.conv4(self.enc_3)))

        return self.enc_4
    
    def decoder(self, x):
        x = self.mish(self.deconv_ln_1(self.deconv1(x)))

        x = self.mish(self.deconv_ln_2(self.deconv2(x)))

        x = self.mish(self.deconv_ln_3(self.deconv3(x)))

        x = self.mish(self.deconv_ln_4(self.deconv4(x)))
        x = self.deconv5(x)

        return x


class Voice_encoder_freq_conv_ver8(nn.Module):
    def __init__(self, in_channels=1, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), fc_size=1024, n_output=256, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # encoder
        self.conv1 = nn.Conv2d(1, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=self.padding, stride=stride)
        self.conv4 = nn.Conv2d(512, 1, kernel_size=kernel_size, padding=self.padding, stride=stride)

        self.conv_ln_1 = nn.LayerNorm([512, 128, 80])
        self.conv_ln_2 = nn.LayerNorm([512, 128, 80])
        self.conv_ln_3 = nn.LayerNorm([512, 128, 80])
        self.conv_ln_4 = nn.LayerNorm([1, 128, 80])

        # decoder
        self.deconv1 = nn.ConvTranspose2d(1, 512, kernel_size=(1, 5), stride=1, padding=(0, 2), output_padding=(0, 0))
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=1, padding=(0, 2), output_padding=(0, 0))
        self.deconv3 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=1, padding=(0, 2), output_padding=(0, 0))
        self.deconv4 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 5), stride=1, padding=(0, 2), output_padding=(0, 0))
        self.deconv5 = nn.Conv2d(512, 1, kernel_size=(1, 5), padding=(0, 2))

        self.deconv_ln_1 = nn.LayerNorm([512, 128, 80])
        self.deconv_ln_2 = nn.LayerNorm([512, 128, 80])
        self.deconv_ln_3 = nn.LayerNorm([512, 128, 80])
        self.deconv_ln_4 = nn.LayerNorm([512, 128, 80])

        # 活性化関数
        self.mish = nn.Mish()

        # 重み初期化
        # encoder
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)

        # decoder
        nn.init.kaiming_normal_(self.deconv1.weight)
        nn.init.kaiming_normal_(self.deconv2.weight)
        nn.init.kaiming_normal_(self.deconv3.weight)
        nn.init.kaiming_normal_(self.deconv4.weight)
        nn.init.kaiming_normal_(self.deconv5.weight)
        nn.init.zeros_(self.deconv1.bias)
        nn.init.zeros_(self.deconv2.bias)
        nn.init.zeros_(self.deconv3.bias)
        nn.init.zeros_(self.deconv4.bias)
        nn.init.zeros_(self.deconv5.bias)


    def forward(self, x):
        x = self.encoder(x)
        dec_x = self.decoder(x)

        return x, dec_x
    
    def encoder(self, x):
        self.enc_1 = self.mish(self.conv_ln_1(self.conv1(x)))
        self.enc_2 = self.mish(self.conv_ln_2(self.conv2(self.enc_1)))
        self.enc_3 = self.mish(self.conv_ln_3(self.conv3(self.enc_2)))
        self.enc_4 = self.mish(self.conv_ln_4(self.conv4(self.enc_3)))

        return self.enc_4
    
    def decoder(self, x):
        x = self.mish(self.deconv_ln_1(self.deconv1(x)))

        x = self.mish(self.deconv_ln_2(self.deconv2(x)))

        x = self.mish(self.deconv_ln_3(self.deconv3(x)))

        x = self.mish(self.deconv_ln_4(self.deconv4(x)))
        x = self.deconv5(x)

        return x

class Voice_encoder_freq_conv_ver9(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, gate_channels=128,kernel=(5, 5), layers=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.kernel = kernel

        # 畳み込み
        self.conv_layers = nn.ModuleList()
        for layer in range(2):
            if layer == 0:
                conv = module.CausalConv2dBlock(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel,
                    stride=(1, 2),
                )
            else:
                conv = module.CausalConv2dBlock(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel,
                    stride=(1, 2)
                )
            
            res = module.Resblock2d(
                residual_chanels=self.out_channels,
                gate_channels=self.gate_channels,
                kernel_size=kernel,
                stride=(1, 1),
                dilation=1
            )
            self.conv_layers.append(conv)
            self.conv_layers.append(res)
        
        # 転置畳み込み
        self.convtrans_layers = nn.ModuleList()
        for layer in range(2):
            if layer / 2 != 0:
                conv = module.CausalConvTranspose2dBlock(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=self.kernel,
                    stride=(1, 2),
                    padding=(2, 0),
                    output_padding=(0, 1)
                )
            else:
                conv = module.CausalConvTranspose2dBlock(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=self.kernel,
                    stride=(1, 2),
                    padding=(2, 0),
                    output_padding=(0, 1)
                )
            self.convtrans_layers.append(conv)
        
        # 転置畳み込み後に畳み込み
        self.convtrans_layers.append(
            module.CausalConv2dBlock(
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=self.kernel,
                stride=(1, 1),
                pad_mode=True
            )
        )
        
    def forward(self, x):
        for f in self.conv_layers:
            x = f(x)
            # print(x.shape)
        v_f = x
        for f in self.convtrans_layers:
            x = f(x)
            # print(x.shape)
        v_c = x
        
        return v_f, v_c
    
    def decoder(self, x):
        for f in self.convtrans_layers:
            x = f(x)
        v_c = x

        return v_c


class WaveNet(nn.Module):
    def __init__(
        self,
        out_channels=80, # 出力のチャネル数
        layers=6, # レイヤー数
        stacks=3, # 畳み込みブロックの数
        residual_channels=64, # 残差結合のチャネル数
        gate_channels=128, # ゲートのチャネル数
        # skip_out_channels=64, # スキップ接続のチャネル数
        kernel_size=2, # 1 次元畳み込みのカーネルサイズ
    ):
        super().__init__()
        self.out_channels = out_channels
        self.first_conv = nn.Conv1d(out_channels, residual_channels, kernel_size=1, padding=0, dilation=1, bias=True)        
        
        # メインとなる畳み込み層
        self.main_conv_layers = nn.ModuleList()
        layers_per_stack = layers // stacks
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = module.ResBlock(
                residual_channels,
                gate_channels,
                kernel_size,
                dilation=dilation,
            )
            self.main_conv_layers.append(conv)
        
        # 波形への変換
        self.last_conv_layers = nn.ModuleList(
            [
                nn.Mish(),
                module.Conv1d1x1(residual_channels, residual_channels),
                nn.Mish(),
                module.Conv1d1x1(residual_channels, out_channels),
            ]
        )

    def forward(self, x):
        x = self.first_conv(x)

        # メインの畳み込み層の処理
        for f in self.main_conv_layers:
            conv_x = f(x)
        
        # 出力を計算
        for f in self.last_conv_layers:
            x = f(conv_x)
        
        return conv_x, x

class WN(nn.Module):
    def __init__(
        self,    
    ):
        pass


if __name__ == '__main__':
    x = torch.randn([1, 1, 128, 80])
    model = Voice_encoder_freq_conv_ver9()
    print(model.modules)
    # print(model.state_dict())
    x = model(x)
