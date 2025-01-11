import numpy as np
import librosa
import pyworld as pw
import os
import shutil
import utils.config as config
import cv2
import utils.functions as functions
import matplotlib.pyplot as plt
from pathlib import Path

dataset_path = Path('dataset/voice_dataset')
voiced_wav = 'wav24kHz16bit'
voiced_npy = 'npy'
voiced_lab = 'lab'
voiceless_wav = 'wav24kHz16bit'
voiceless_npy = 'npy'
voiceless_lab = 'lab'

phonemedict = functions.generate_phoneme_dict()

for i in range(1, 101):
    if dataset_path.joinpath('jvs'+str(i).zfill(3), voiced_wav).exists():
        print(i)
        # wav
        voiced_wav_path = dataset_path.joinpath('jvs'+str(i).zfill(3), voiced_wav)
        voiceless_wav_path = dataset_path.joinpath('jvs'+str(i).zfill(3)+'_w', voiceless_wav)

        # lab
        voiced_lab_path = dataset_path.joinpath('jvs'+str(i).zfill(3), voiced_lab)
        voiceless_lab_path = dataset_path.joinpath('jvs'+str(i).zfill(3), voiceless_lab)

        voiced_out_path = dataset_path.joinpath('jvs'+str(i).zfill(3), voiced_npy)
        voiceless_out_path = dataset_path.joinpath('jvs'+str(i).zfill(3)+'_w', voiceless_npy)
        
        for file in voiced_wav_path.iterdir():
            # 音声波形読み込み                                                                     
            wav,sr = librosa.load(file,sr=config.sr)                                                                                            
            #短時間フーリエ変換
            D = librosa.stft(y=wav, n_fft = config.n_fft, hop_length = config.hop_length, win_length = config.win_length, pad_mode='reflect').T
            #振幅スペクトル　位相スペクトル　抽出
            sp,phase = librosa.magphase(D)    
            #メルスペクトルを抽出
            msp = np.matmul(sp, functions.mel_filter)
            #対数をとる
            lmsp = functions.dynamic_range_compression(msp)
            # 基本周波数、ケプストラム、非周期性指標を取得
            f0, cp, ap = pw.wav2world(wav.astype(np.float64), sr, fft_size=config.n_fft, frame_period=config.hop_length*1000/config.sr)
            # 計算の内容
            print(f0.reshape(-1,1).shape)
            print(functions.mel_freqs.shape)
            f0dif = f0.reshape(-1,1)*config.f0_scale - functions.mel_freqs
            f0mat = np.eye(config.n_mels)[(f0dif**2).argmin(axis=1)].astype(np.float32)                                                                     #f0mat
            f0mat[:,0] = 0

            # 連続基本周波数
            cf0 = functions.interp1d(f0, kind='linear')
            # cf0 = cf0[:, np.newaxis] if len(cf0.shape) == 1 else cf0

            cf0dif = cf0.reshape(-1,1)*config.f0_scale - functions.mel_freqs
            cf0mat = np.eye(config.n_mels)[(cf0dif**2).argmin(axis=1)].astype(np.float32)

            # # 有声無声フラグ
            vuv = (f0 > 0).astype(np.float32)
            # vuv = vuv[:, np.newaxis] if len(vuv.shape) == 1 else vuv

            # cf0v = np.hstack([cf0, vuv])

            # 帯域非周期性指標を抽出
            cap = pw.code_aperiodicity(ap,fs=config.sr)                                                                                                     #ケプストラム
            cap = np.exp(cap)                                                                                                                               #　eの乗数
            
            # スペクトル包絡を抽出
            mcp = pw.code_spectral_envelope(cp,config.sr,config.n_mels)   
            
            # ppgmat
            ppg = np.zeros(msp.shape[0])
            if Path.exists(voiced_lab_path.joinpath(file.name)):
                label = open(voiced_lab_path.joinpath(file.name))
                for line in label.readlines():
                    labs = line.split(' ')
                    start = int(float(labs[0]) * config.sr / config.hop_length)
                    ppg[start:] = phonemedict[labs[2].replace('\n', '')]
                label.close()
                voiced_ppgmat = np.eye(36)[np.array(ppg).astype(np.uint8)].astype(np.float32)                                                                                  #メルケプストラム

            #voiceless
            file_pathw = voiceless_wav_path.joinpath(file.name)
            wav, sr = librosa.load(file_pathw,sr=config.sr)
            D = librosa.stft(y=wav, n_fft = config.n_fft, hop_length = config.hop_length, win_length =config.win_length, pad_mode='reflect').T
            sp,phase = librosa.magphase(D)
            mspw = np.matmul(sp, functions.mel_filter)
            lmspw = functions.dynamic_range_compression(mspw)

            if Path.exists(voiceless_lab_path.joinpath(file.name)):
                label = open(voiceless_lab_path.joinpath(file.name))
                for line in label.readlines():
                    labs = line.split(' ')
                    start = int(float(labs[0]) * config.sr / config.hop_length)
                    ppg[start:] = phonemedict[labs[2].replace('\n', '')]
                label.close()
                voiceless_ppgmat = np.eye(36)[np.array(ppg).astype(np.uint8)].astype(np.float32)

            if lmsp.shape[0] != lmspw.shape[0]:
                lmspw = lmspw[:lmsp.shape[0]]
            if Path.exists(voiceless_lab_path.joinpath(file.name)):
                if voiced_ppgmat.shape[0] != voiceless_ppgmat.shape[0]:
                    voiceless_ppgmat = voiceless_ppgmat[:voiced_ppgmat.shape[0]]

            if lmsp.shape[0] != lmspw.shape[0]:
                print(f'lmsp {lmsp.shape[0]}, lmspw {lmspw.shape[0]}')
            
            if Path.exists(voiceless_lab_path.joinpath(file.name)):
                if voiced_ppgmat.shape[0] != voiceless_ppgmat.shape[0]:
                    print(f'voice_ppg {voiced_ppgmat.shape} voiceless_ppg {voiceless_ppgmat.shape}')
            
            if Path.exists(voiceless_lab_path.joinpath(file.name)):
                if lmsp.shape[0] != voiced_ppgmat.shape[0]:
                    print('not same size')
            
            # folder作成
            voiceless_out_path.mkdir(exist_ok=True, parents=True)
            voiced_out_path.mkdir(exist_ok=True, parents=True)

            # np.save(voiceless_out_path.joinpath(file.name.replace('.wav', '_msp.npy')), lmspw)
            # np.save(voiced_out_path.joinpath(file.name.replace('.wav','_msp.npy')),lmsp)
            # np.save(voiced_out_path.joinpath(file.name.replace('.wav','_f0.npy')),f0)
            # np.save(voiceless_out_path.joinpath((file.name).replace('.wav','_f0.npy')),f0)
            # np.save(voiced_out_path.joinpath((file.name).replace('.wav','_f0mat.npy')), f0mat)
            # np.save(voiceless_out_path.joinpath((file.name).replace('.wav','_f0mat.npy')), f0mat)
            # np.save(voiced_out_path.joinpath((file.name).replace('.wav','_cap.npy')), cap)
            # np.save(voiceless_out_path.joinpath((file.name).replace('.wav','_cap.npy')), cap)
            # np.save(voiced_out_path.joinpath((file.name).replace('.wav','_mcp.npy')), mcp)
            # np.save(voiceless_out_path.joinpath((file.name).replace('.wav','_mcp.npy')), mcp)
            # if Path.exists(voiceless_lab_path.joinpath(file.name)):
            #     np.save(voiced_out_path.joinpath((file.name).replace('.wav','_ppgmat.npy')), voiced_ppgmat)
            #     np.save(voiceless_out_path.joinpath((file.name).replace('.wav','_ppgmat.npy')), voiceless_ppgmat)
            # np.save(voiced_out_path.joinpath((file.name).replace('.wav','_cf0.npy')), cf0)
            # np.save(voiceless_out_path.joinpath((file.name).replace('.wav','_cf0.npy')), cf0)
            # np.save(voiceless_out_path.joinpath((file.name).replace('.wav','_cf0mat.npy')), cf0mat)
            # np.save(voiced_out_path.joinpath((file.name).replace('.wav','_cf0mat.npy')), cf0mat)
                
print("finesh")
