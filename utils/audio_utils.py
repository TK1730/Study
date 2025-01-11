import torch
import numpy as np
import torchaudio
import librosa
from librosa.filters import mel as librosa_mel_fn

from utils import config, functions

# librosaでの音声データ処理
def spec_2_melspec(
    spec,
    sr,
    n_mels,
    n_fft,
    fmin,
    fmax,
    **kwargs,    
    ):
    mel_scale = librosa_mel_fn(
        sr=sr,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
        n_mels=n_mels,
        htk=False,
        norm='slaney'
    ).T
    msp = np.matmul(spec, mel_scale)
    return msp


# pytorchでの音声データ処理
def spec_2_melspec_torch(
        spec: torch.Tensor,
        sr: int,
        n_mels: int,
        n_fft: int,
        fmin,
        fmax,
        **kwargs):
    # メルフィルタバンクを生成
    mel_scale = librosa_mel_fn(
        sr=sr,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
        n_mels=n_mels,
        htk=False,
        norm='slaney').T
    mel_scale = torch.from_numpy(mel_scale).to(dtype=spec.dtype, device=spec.device)
    # メルスペクトログラムに変換
    msp = torch.matmul(spec, mel_scale)
    return msp

if __name__ == '__main__':
    wav_path = 'dataset/jvs_ver1/jvs001/nonpara30/wav24kHz16bit/BASIC5000_0025.wav'
    wav, sr = librosa.load(wav_path, sr=config.sr)
    D = librosa.stft(y=wav, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length, pad_mode='reflect').T
    sp, phase = librosa.magphase(D)
    sp_torch = torch.from_numpy(sp)

    # librosa mel
    lib_mel = spec_2_melspec(sp, sr=config.sr, n_mels=config.n_mels, n_fft=config.n_fft, fmin=config.fmin, fmax=config.fmax)
    print(f'lib_mel: {lib_mel.dtype}')
    # torch mel
    tor_mel = spec_2_melspec_torch(sp_torch, sr=config.sr, n_mels=config.n_mels, n_fft=config.n_fft, fmin=config.fmin, fmax=config.fmax)
    print(f'tor_mel: {tor_mel.dtype}')
    lib_mel = torch.from_numpy(lib_mel).to(dtype=tor_mel.dtype)

