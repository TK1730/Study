import numpy as np
import librosa
import pyworld as pw
import os
import shutil
import utils.config as config
import cv2
import matplotlib.pyplot as plt

clear_data = False

dataset_path = 'dataset/jvs_ver1_exp6/'

voiced_wav_path = '/nonpara30/wav24kHz16bit/'
voiced_npy_path = '/nonpara30/npy/'
voiced_lab_path = '/nonpara30/lab/mon/'

voiceless_wav_path = '/nonpara30w/wav24kHz16bit/'
voiceless_npy_path = '/nonpara30w/npy/'


def generate_phoneme_dict():
    phonemelist = []

    f = open('phoneme.txt')                     #フォンテキストを開く
    for phoneme in f.readlines():
        phonemelist = phoneme.replace("'","").split(', ')
    f.close()
    phonemedict = {p: phonemelist.index(p) for p in phonemelist}
    phonemedict['a:'] = phonemedict['a'] # adhoc
    phonemedict['i:'] = phonemedict['i'] # adhoc
    phonemedict['u:'] = phonemedict['u'] # adhoc
    phonemedict['e:'] = phonemedict['e'] # adhoc
    phonemedict['o:'] = phonemedict['o'] # adhoc
    phonemedict['A'] = phonemedict['a'] # adhoc
    phonemedict['I'] = phonemedict['i'] # adhoc
    phonemedict['U'] = phonemedict['u'] # adhoc
    phonemedict['E'] = phonemedict['e'] # adhoc
    phonemedict['O'] = phonemedict['o'] # adhoc
    phonemedict['pau'] = phonemedict['sil'] # adhoc
    phonemedict['silB'] = phonemedict['sil'] # adhoc
    phonemedict['silE'] = phonemedict['sil'] # adhoc
    phonemedict['q'] = phonemedict['sil']
    phonemedict['sp'] = phonemedict['sil'] # adhoc
    phonemedict['cl'] = phonemedict['sil']

    return phonemedict

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(x.clip(clip_val,None))

def dynamic_range_decompression(x, C=1):
    return np.exp(x)

phonemedict = generate_phoneme_dict()

mel_freqs = librosa.mel_frequencies(n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax, htk=False).reshape(1,-1)
mel_filter = librosa.filters.mel(sr=config.sr, n_fft=config.n_fft, fmin=config.fmin, fmax=config.fmax, n_mels=config.n_mels, htk=False, norm='slaney').T

for person in os.listdir(dataset_path):
    if os.path.isdir(dataset_path+person):
        print(person+'>>>>>>>>>>>>>>>')
        voiced_out_path = dataset_path + person + voiced_npy_path
        voiceless_out_path = dataset_path + person + voiceless_npy_path

        if clear_data:
            if os.path.exists(voiced_out_path):
                shutil.rmtree(voiced_out_path)
            if os.path.exists(voiceless_out_path):
                shutil.rmtree(voiceless_out_path)

        #os.makedirs(voiced_out_path,exist_ok=True)
        #os.makedirs(voiceless_out_path,exist_ok=True)


        for f in os.listdir(dataset_path + person + voiced_wav_path):
            #voiced
            file_path = dataset_path + person + voiced_wav_path + f
            label_path = dataset_path + person + voiced_lab_path + f.replace('.wav','.lab')                                                                     
            wav,sr = librosa.load(file_path,sr=config.sr)                                                                                                   #音声読み込み
            D = librosa.stft(y=wav, n_fft = config.n_fft, hop_length = config.hop_length, win_length = config.win_length, pad_mode='reflect').T             #短時間フーリエ変換
            sp,phase = librosa.magphase(D)                                                                                                                  #振幅スペクトル　位相スペクトル　抽出    
            msp = np.matmul(sp,mel_filter)                                                                                                                  #メルスペクトルを抽出
            lmsp = dynamic_range_compression(msp)                                                                                                           #対数をとる                                                                                                        #メルスペクトル抽出

            f0, cp, ap = pw.wav2world(wav.astype(np.float64), sr,frame_period=config.hop_length*1000/config.sr) 
            
            # 有声,無声フラグ
            vuv = (f0 > 0).astype(np.float32)
            f0dif = f0.reshape(-1,1)*config.f0_scale - mel_freqs
            f0mat = np.eye(config.n_mels)[(f0dif**2).argmin(axis=1)].astype(np.float32)
            f0mat[:, 0] = 0 

            print('cp_fft_size',int(cp.shape[1] - 1)*2)

            print('ap_fft_size',int(ap.shape[1] - 1)*2)

            cap = pw.code_aperiodicity(ap,fs=config.sr)                                                                                                     #ケプストラム
            cap = np.exp(cap)                                                                                                                               #　eの乗数

            mcp = pw.code_spectral_envelope(cp,config.sr,config.n_mels)                                                                                     #メルケプストラム

            if os.path.exists(label_path):
                ppg = np.zeros(msp.shape[0])
                label = open(label_path)
                for line in label.readlines():
                    labs = line.split(' ')
                    start = int(float(labs[0])*config.sr/config.hop_length) #1/200 = 80* 1/16000
                    ppg[start:] = phonemedict[labs[2].replace('\n', '')]
                label.close()
                ppgmat = np.eye(36)[np.array(ppg).astype(np.uint8)].astype(np.float32)                                                                      # ppgmat ppg の違いがわかんね

            #voiceless
            file_pathw = dataset_path + person + voiceless_wav_path + f
            wav,sr = librosa.load(file_pathw,sr=config.sr)
            D = librosa.stft(y=wav, n_fft = config.n_fft, hop_length = config.hop_length, win_length =config.win_length, pad_mode='reflect').T
            sp,phase = librosa.magphase(D)
            mspw = np.matmul(sp,mel_filter)
            lmspw = dynamic_range_compression(mspw)

            if lmsp.shape[0] != lmspw.shape[0]:
                lmspw = lmspw[:msp.shape[0]]
            
            print(lmsp.shape,lmspw.shape,"f0",f0.shape,'mcp',mcp.shape,'cap',cap.shape)
            
            # np.save(voiceless_out_path+f.replace('.wav','_msp.npy'),lmspw)
            # np.save(voiced_out_path+f.replace('.wav','_msp.npy'),lmsp)
            # np.save(voiceless_out_path+f.replace('.wav','_f0.npy'),f0)
            # np.save(voiceless_out_path+f.replace('.wav','_f0mat.npy'),f0mat)
            # np.save(voiced_out_path+f.replace('.wav','_mcp.npy'),mcp)
            # np.save(voiced_out_path+f.replace('.wav','_cap.npy'),cap)
            # if os.path.exists(label_path):
            #     np.save(voiced_out_path+f.replace('.wav','_ppg.npy'),ppg)
            #     np.save(voiced_out_path+f.replace('.wav','_ppgmat.npy'),ppgmat)

            

print("finesh")
