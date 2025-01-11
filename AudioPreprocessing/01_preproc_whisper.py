import numpy as np
import librosa
import pyworld as pw
import os
import utils.config as config
import shutil
import utils.functions as functions
from pathlib import Path

dataset_path = Path('dataset/voice_dataset')
voice_wav_path = 'wav24kHz16bit'
voice_npy_path = 'npy'
voice_label_path = 'lab'

mel_freqs = librosa.mel_frequencies(n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax, htk=False).reshape(1,-1)
mel_filter = librosa.filters.mel(sr=config.sr, n_fft=config.n_fft, fmin=config.fmin, fmax=config.fmax, n_mels=config.n_mels, htk=False, norm='slaney').T

# データ削除させるかどうか
clear_data = True

phonemedict = functions.generate_phoneme_dict()

for i, person in enumerate(dataset_path.iterdir()):

    # 音声ファイル取得
    voice_wav = person.joinpath(voice_wav_path)
    voice_wav = sorted(voice_wav.iterdir())
    # npyファイル取得
    voice_npy = person.joinpath(voice_npy_path)
    # labelファイル取得

    voice_label = person.joinpath(voice_label_path)
    if voice_label.is_dir():
        voice_label = sorted(voice_label.iterdir())
        for data in voice_wav:
            # 音声読み込み
            wav, sr = librosa.load(data, sr=config.sr)
            D = librosa.stft(y=wav, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length, pad_mode='reflect').T
            # 振幅スペクトル　位相スペクトル　抽出
            sp, phase = librosa.magphase(D)    
            # メルスペクトルを抽出
            msp = np.matmul(sp, mel_filter)
            # 対数をとる
            lmsp = functions.dynamic_range_compression(msp)
            # 基本周波数、ケプストラム、非周期性指標を取得
            f0, cp, ap = pw.wav2world(wav.astype(np.float64), sr, frame_period=config.hop_length * 1000 / config.sr)
            # 計算の内容
            f0dif = f0.reshape(-1, 1) * config.f0_scale - mel_freqs
            f0mat = np.eye(config.n_mels)[(f0dif**2).argmin(axis=1)].astype(np.float32)                                                                     #f0mat
            f0mat[:, 0] = 0

            ppg = np.zeros(msp.shape[0])

            for lab in voice_label:
                if Path.exists(lab):
                    if data.stem in str(lab):
                        label = open(lab)
                        for line in label.readlines():
                            labs = line.split(' ')
                            start = int(float(labs[0]) * config.sr / config.hop_length)
                            ppg[start:] = phonemedict[labs[2].replace('\n', '')]
                        label.close()
                        ppgmat = np.eye(36)[np.array(ppg).astype(np.uint8)].astype(np.float32)
                        
            if ppgmat.shape[0] != lmsp.shape[0]:
                ppgmat = ppgmat[:lmsp.shape[0]]

            if lmsp.shape[0] != ppgmat.shape[0]:
                lmsp = lmsp[:ppgmat.shape[0]]
            
            if not voice_npy.exists():
                voice_npy.mkdir(parents=True)

            np.save(str(voice_npy) + '/' + data.stem + '_ppgmat.npy', ppgmat)
            np.save(str(voice_npy) + '/' + data.stem + '_msp.npy', lmsp)
    
print('finish')
