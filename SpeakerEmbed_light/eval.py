import sys
sys.path.append('./')
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import cv2
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pyworld as pw
import soundfile as sf
import pickle

from net.models import VocalNet
from dataloader.clip_dataset import Clip_dataset
from pathlib import Path
from net.cnn_custom import Voice_encoder_freq_conv_ver9


def Padding(x, frame_length=128):
    pad = x.shape[0] % frame_length
    if pad != 0:
        x = np.pad(x, ((0, frame_length - pad), (0, 0)))
    x = x.reshape(x.shape[0]//frame_length, 1, frame_length, 80)
    over_pad = frame_length - pad
    return x, over_pad

def Msp2Wav(x):
    # log 2 liner
    x = functions.dynamic_range_decompression(x)
    # mel2stft
    x = librosa.feature.inverse.mel_to_stft(M=x.T, sr=config.sr, n_fft=config.n_fft, power=1)
    # 位相復元
    x = librosa.griffinlim(x, n_iter=10000, hop_length=config.hop_length, n_fft=config.n_fft)
    return x

# setting
epochs = 20000
test_rate = 0.2
batch_size = 10
accumulation_step = 5
frame_length = 128
input_type = ['mcp']
t = 50
eps = 1e-5

in_channels = 1
hidden_channels = 256
inter_channels = 128
# dec
encoder_dwn_kernel_size = [[5, 4], [5, 4]]
encoder_resblock_kernel_size = [[3, 3], [3, 3], [3, 3],]
encoder_stride_size = [[1, 2], [1, 2]]
encoder_dilation_rate = 2
encoder_resblock_layers = 3
encoder_p_dropout = 0
# deco
resblock_kernel_sizes = [[3], [7], [11]]
resblock_dilation_sizes = [[[1, 1], [3, 1], [5, 1]],
                           [[1, 1], [3, 1], [5, 1]],
                           [[1, 1], [3, 1], [5, 1]]]
upsample_rates = [[1, 2], [1, 2],]
upsample_initial_channel = 512
upsample_kernel_sizes = [[5, 4],[5, 4],]
decoder_LayerNorm = True

# gpu確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# モデル, 最適化関数の定義
def create_model():
    return VocalNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            inter_channels=inter_channels,
            encoder_dwn_kernel_size=encoder_dwn_kernel_size,
            encoder_resblock_kernel_size=encoder_resblock_kernel_size,
            encoder_stride_size=encoder_stride_size,
            encoder_dilation_rate=encoder_dilation_rate,
            encoder_resblock_layers=encoder_resblock_layers,
            encoder_LayerNorm=True,
            encoder_p_dropout=encoder_p_dropout,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_kernels=upsample_kernel_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            decoder_LayerNorm=True
            ).to(device)

def load_use_data(fpath:Path, fname):
    with open(fpath.joinpath(fname+'.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data

def load_model(model:nn.Module, fpath:Path, fname):
    weights = torch.load(fpath.joinpath(fname))
    model.load_state_dict(weights)
    return model

def data_load(voice_path, whisp_path, idx):
    vsp_file_index = np.random.randint(0, len(voice_path[idx]))
    wsp_file_index = np.random.randint(0, len(whisp_path[idx]))
    vsp = voice_path[idx][vsp_file_index]
    wsp = whisp_path[idx][wsp_file_index]

    vsp = np.load(vsp).astype(np.float32)
    wsp = np.load(wsp).astype(np.float32)


    if frame_length > 0:
        if frame_length >= vsp.shape[0]:
            vsp = np.pad(vsp,[(0,frame_length+1-vsp.shape[0]),(0,0)], mode='constant')
        st = np.random.randint(0, vsp.shape[0]-frame_length)
        vsp = vsp[st:st+frame_length]
            
        if frame_length >= wsp.shape[0]:
            wsp = np.pad(wsp,[(0,frame_length+1-wsp.shape[0]),(0,0)], mode='constant')
        st = np.random.randint(0, wsp.shape[0]-frame_length)
        wsp = wsp[st:st+frame_length]
    
    vsp = vsp.reshape(1, frame_length, 80)
    wsp = wsp.reshape(1, frame_length, 80)

    return vsp, wsp

def cos_sim_show(logits, winname, height=500):
    logits = logits.to('cpu').detach().numpy()
    logits = (logits/t + 1.0) / 2.0
    logits = cv2.resize(logits, dsize=(height, height), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(winname, logits)
    cv2.waitKey(0)

def eval(pseud_model, whisp_model,voice_train, voice_test, whisp_train, whisp_test):
    # 推論モード
    if pseud_model.training:
        pseud_model.eval()
        whisp_model.eval()
        print("現在のモードは評価モードです")
    else:
        print("現在のモードは評価モードです")
    
    for i in range(0, 10):
        voice, whisp = data_load(voice_test, whisp_test, i)
        if i == 0:
            v = voice
            w = whisp
        else:
            v = np.vstack([v, voice])
            w = np.vstack([w, whisp])
    
    v = torch.from_numpy(v).reshape(-1, 1, frame_length, 80).to(device)
    w = torch.from_numpy(w).reshape(-1, 1, frame_length, 80).to(device)

    v_f, v_c = pseud_model(v)
    print('ひとつめ終わり')
    w_f, w_c = whisp_model(w)
    print('二つ目終わり')
    v_f = torch.mean(v_f, dim=(1, 2))
    w_f = torch.mean(w_f, dim=(1, 2))
    vf_l2 = torch.sqrt((v_f**2).sum(dim=1))
    v_f = (v_f.T/(vf_l2+eps)).T
    wf_l2 = torch.sqrt((w_f**2).sum(dim=1))
    w_f = (w_f.T/(wf_l2+eps)).T

    print('pass')
    # 類似度計算
    logits = torch.matmul(v_f, w_f.T) * t
    labels = torch.arange(0, v.size(0), dtype=torch.long, device=device)
    cos_sim_show(logits, 'cos_sim')




if __name__ == '__main__':
    # モデルパス
    save_folder = Path("SpeakerEmbed_light").joinpath("20241117")
    # モデル作成と重み読み込み
    pseud_model = create_model()
    whisp_model = create_model()
    pseud_model = load_model(pseud_model, save_folder, "pseudo_encoder_bestloss.pth")
    whisp_model = load_model(whisp_model, save_folder, "whisp_encoder_bestloss.pth")

    # 使用したデータ読み込み
    voice_train = load_use_data(save_folder, 'voice_train')
    voice_test = load_use_data(save_folder, 'voice_test')
    whisp_train = load_use_data(save_folder, 'whisp_train')
    whisp_test = load_use_data(save_folder, 'whisp_test')

    eval(pseud_model, whisp_model, voice_train, voice_test, whisp_train, whisp_test)
