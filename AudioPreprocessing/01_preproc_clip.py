import sys
sys.path.append('./')
import numpy as np
import librosa
import pyworld as pw
import os
import utils.config as config
import shutil
import utils.functions as functions
import soundfile as sf
from pathlib import Path
from utils import config, functions

# save
save_path = Path('dataset/jvs_ver3')
# load
dataset_path = Path('dataset/jvs_ver3')

def make_npy(file_type, save_path):
    for person in dataset_path.iterdir():
        folder = person.joinpath(file_type, 'wav24kHz16bit')
        print(person.name)
        for data in folder.iterdir():
            save = save_path.joinpath(person.stem, file_type, 'wav24kHz16bit')
            if not save.exists():
                save.mkdir(parents=True)

            save_file = save.joinpath(data.name)
            # shutil.copy(str(data), str(save_file))
            # save_wav = functions.ludness_normalize(save_file, config.lufs_throat)
            # sf.write(save_file, save_wav, samplerate=config.sr)

            # データ読み込み
            wav, sr = librosa.load(save_file, sr=config.sr)
            # 短時間フーリエ変換
            D = librosa.stft(wav, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length, pad_mode='reflect').T
            # 振幅スペクトル　位相スペクトル　抽出
            sp, phase = librosa.magphase(D)
            # メルスペクトルを抽出
            msp = np.matmul(sp, functions.mel_filter)
            # 対数をとる
            lmsp = functions.dynamic_range_compression(msp)
            # 基本周波数、ケプストラム、非周期性指標を取得
            f0, cp, ap = pw.wav2world(wav.astype(np.float64), sr, fft_size=config.n_fft, frame_period=config.hop_length*1000/config.sr)
            # スペクトル包絡を抽出
            mcp = pw.code_spectral_envelope(cp, config.sr, config.n_mels)

            save_npy = save_path.joinpath(person.stem, file_type, 'npy')
            if not save_npy.exists():
                save_npy.mkdir(parents=True)
            save_npy_file = save_npy.joinpath(save_file.stem)
            # np.save(str(save_npy_file) + '_msp.npy', lmsp)
            # np.save(str(save_npy_file) + '_mcp.npy', mcp)
            np.save(str(save_npy_file) + '_cp.npy', cp)


if __name__ == '__main__':
    make_npy("nonpara30w_mean", save_path)
    make_npy("whisper10", save_path)
        
# for person in dataset_path.iterdir():
#     folder = person.joinpath('whisper10', 'wav24kHz16bit')
#     print(person.name)
#     for data in folder.iterdir():
#         save = save_path.joinpath(person.stem, 'whisper10', 'wav24kHz16bit')
#         if not save.exists():
#             save.mkdir(parents=True)

#         save_file = save.joinpath(data.name)
#         # shutil.copy(str(data), str(save_file))
#         # save_wav = functions.ludness_normalize(save_file, config.lufs_throat)
#         # sf.write(save_file, save_wav, samplerate=config.sr)
#         # データ読み込み
#         wav, sr = librosa.load(save_file, sr=config.sr)
#         # 短時間フーリエ変換
#         D = librosa.stft(wav, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length, pad_mode='reflect').T
#         # 振幅スペクトル　位相スペクトル　抽出
#         sp, phase = librosa.magphase(D)
#         # メルスペクトルを抽出
#         msp = np.matmul(sp, functions.mel_filter)
#         # 対数をとる
#         lmsp = functions.dynamic_range_compression(msp)
#         # 基本周波数、ケプストラム、非周期性指標を取得
#         f0, cp, ap = pw.wav2world(wav.astype(np.float64), sr, fft_size=config.n_fft, frame_period=config.hop_length*1000/config.sr)
#         # スペクトル包絡を抽出
#         mcp = pw.code_spectral_envelope(cp, config.sr, config.n_mels)

#         save_npy = save_path.joinpath(person.stem, 'whisper10', 'npy')
#         if not save_npy.exists():
#             save_npy.mkdir(parents=True)
#         save_npy_file = save_npy.joinpath(save_file.stem)
        
#         np.save(str(save_npy_file) + '_msp.npy', lmsp)
#         np.save(str(save_npy_file) + '_mcp.npy', mcp)

# for person in dataset_path.iterdir():
#     folder = person.joinpath('nonpara30w_mean', 'wav24kHz16bit')
#     print(person.name)
#     for data in folder.iterdir():
#         save = save_path.joinpath(person.stem, 'nonpara30w_mean', 'wav24kHz16bit')
#         if not save.exists():
#             save.mkdir(parents=True)

#         save_file = save.joinpath(data.name)
#         # shutil.copy(str(data), str(save_file))
#         # save_wav = functions.ludness_normalize(save_file, config.lufs_throat)
#         # sf.write(save_file, save_wav, samplerate=config.sr)

#         # データ読み込み
#         wav, sr = librosa.load(save_file, sr=config.sr)
#         # 短時間フーリエ変換
#         D = librosa.stft(wav, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length, pad_mode='reflect').T
#         # 振幅スペクトル　位相スペクトル　抽出
#         sp, phase = librosa.magphase(D)
#         # メルスペクトルを抽出
#         msp = np.matmul(sp, functions.mel_filter)
#         # 対数をとる
#         lmsp = functions.dynamic_range_compression(msp)
#         # 基本周波数、ケプストラム、非周期性指標を取得
#         f0, cp, ap = pw.wav2world(wav.astype(np.float64), sr, fft_size=config.n_fft, frame_period=config.hop_length*1000/config.sr)
#         # スペクトル包絡を抽出
#         mcp = pw.code_spectral_envelope(cp, config.sr, config.n_mels)

#         save_npy = save_path.joinpath(person.stem, 'nonpara30w_mean', 'npy')
#         if not save_npy.exists():
#             save_npy.mkdir(parents=True)
#         save_npy_file = save_npy.joinpath(save_file.stem)
#         np.save(str(save_npy_file) + '_msp.npy', lmsp)
#         np.save(str(save_npy_file) + '_mcp.npy', mcp)
