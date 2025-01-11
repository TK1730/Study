import numpy as np

import torch
import torch.nn as nn

# 入力画像のサイズ (例: 1チャネル, 6x6)
input_image = torch.randn(1, 1, 6, 6)

# ConvTranspose2dの設定
conv_transpose = nn.ConvTranspose2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    stride=8,
                                    padding=(3-2)//2,
                                    output_padding=0)

# 逆畳み込みを適用
output_image = conv_transpose(input_image)

# print(f"入力画像のサイズ: {input_image.shape}")
# print(f"出力画像のサイズ: {output_image.shape}")

import torch
import torch.nn.functional as F

# サンプル時系列データ
x = torch.randn(1, 22050)  # バッチサイズ1, チャネル数1, データ長1024

# STFTのパラメータ
n_fft = 1024  # フーリエ変換のサイズ（ウィンドウサイズ）
hop_length = 256  # オーバーラップなしのステップサイズ
window = torch.hann_window(n_fft)  # ハニング窓

# STFTの実行
stft_result = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=1024, window=window, return_complex=True)

print(f"STFTの結果のサイズ: {stft_result.shape}")

print(np.exp(0.07))
