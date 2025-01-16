import torch
import numpy as np
import torchaudio
import librosa
import pyworld as pw
import librosa.filters

from utils import config, functions

# librosaでの音声データ処理
def mel_filter(
        sr: int=config.sr,
        n_fft: int=config.n_fft,
        fmin: int=config.fmin,
        fmax: int=config.fmax,
        n_mels: int=config.n_mels,
        **kwargs):
    """
      メルフィルター作成
    """
    return librosa.filters.mel(sr=sr, n_fft=n_fft, fmin=fmin, fmax=fmax, n_mels=n_mels, **kwargs)


def spec_to_melspec(
    spec: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    fmin: int,
    fmax: int,
    **kwargs,    
    ):
    """spectrogramをmel-scpectrogramに変換

    Args:
        spec (np.ndarray): spectrogram
        sr (int): sampling_rate
        n_mels (int): mel_dim
        n_fft (int): fft_size
        fmin (int): min_frequency
        fmax (int): max_frequency

    Returns:
        mel_spec: mel_spectrogram 
    """
    # メルフィルタバンクを作成
    mel_scale = mel_filter(
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


def spectrum(wav, ):
    pass

def invers_spectrum():
    pass

def spectrogram(
        input: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        center= True,
        pad_mode='const',
        power=1,
):
    window = torch.hann_window(window_length=win_length)
    # 短時間フーリエ変換
    spec = torch.stft(input, n_fft=n_fft, hop_length=hop_length, win_length=win_length, pad_mode=pad_mode, window=window, center=center, return_complex=True)
    spec = torch.abs(spec).T
    spec **= power
    return spec

def invers_spectrogram( 
        spec: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        center: bool=True,
        **kwargs,
):
    window = torch.hann_window(window_length=win_length)
    # 逆短時間フーリエ変換
    spec = torch.istft(spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, **kwargs)
    return spec

def spec_2_melspec_torch(spec: torch.Tensor, sr: int, n_mels: int,
        n_fft: int, fmin: int, fmax: int, **kwargs):
    # メルフィルタバンクを生成
    mel_scale = mel_filter(sr=sr, n_fft=n_fft, fmin=fmin, fmax=fmax,
        n_mels=n_mels, **kwargs).T
    mel_scale = torch.from_numpy(mel_scale).to(dtype=spec.dtype, device=spec.device)
    
    # メルスペクトログラムに変換
    msp = torch.matmul(spec, mel_scale)
    return msp

def cep_to_melcep_torch(
        spec: torch.Tensor,
        sr: int,
        n_mels: int,
        n_fft: int=config.n_fft,
        fmin: int=config.fmin,
        fmax: int=config.fmax,
        **kwargs):
    # メルフィルタバンクを生成
    mel_scale = mel_filter(
        sr=sr,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
        n_mels=n_mels,
        **kwargs).T
    mel_scale = torch.from_numpy(mel_scale).to(dtype=spec.dtype, device=spec.device)

    melcep = torch.matmul()



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    wav_path = 'dataset/jvs_ver1/jvs001/nonpara30/wav24kHz16bit/BASIC5000_0025.wav'
    wav, sr = librosa.load(wav_path, sr=config.sr)

    # 短時間フーリエ変換
    D = librosa.stft(wav, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length, pad_mode='reflect').T
    # 振幅スペクトル　位相スペクトル　抽出
    sp, phase = librosa.magphase(D)
    # メルスペクトルを抽出
    msp = np.matmul(sp, functions.mel_filter)
    msp = msp.clip(0, None)
    print(msp.max())
    print(msp.min())

    mel_scale = mel_filter(
        sr=sr,
        n_fft=config.n_fft,
        n_mels=config.n_mels,).T
    print(mel_scale.shape)
    
    f0, sp, ap = pw.wav2world(wav.astype(np.float64), sr, fft_size=config.n_fft, frame_period=config.hop_length*1000/config.sr)

    mcp = pw.code_spectral_envelope(sp, sr, 80)

    mel_sp = np.matmul(sp, mel_scale)

    mel_lib = librosa.feature.mfcc(y=wav, sr=config.sr, n_mfcc=80, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length, pad_mode='reflect').T
    print(mcp.shape)
    print(mel_sp.shape)

  
