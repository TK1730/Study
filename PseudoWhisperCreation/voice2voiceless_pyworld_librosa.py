import numpy as np
import librosa
import pyworld as pw
import torch
import os
import cv2
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal

dataset_path = 'dataset/jvs_ver1/'
voiced_wav_path = '/nonpara30/wav24kHz16bit/'
voiceless_wav_path = '/whisper10/wav24kHz16bit/'
result_wav_path = '/nonpara30w_mean/wav24kHz16bit/'

high_pass = False

sr = 44100

for person in os.listdir(dataset_path):
    if os.path.isdir(dataset_path+person):
        print(person+'>>>>>>>>>>>>>>>')

        #voiced
        
        for idx, f in enumerate(os.listdir(dataset_path+person+voiced_wav_path)):
            file_path = dataset_path + person + voiced_wav_path + f
            wav,sr = librosa.load(file_path)
            f0, cp, ap = pw.wav2world(wav.astype(np.float64), sr)
            if idx == 0:
                cp_all = cp.T
            else:
                cp_all = np.hstack((cp_all,cp.T))
        
        
        x = cp_all.max(axis=1)
        xmax = x.max()
        x /= xmax

        #voiceless 
        for idx, f in enumerate(os.listdir(dataset_path+person+voiceless_wav_path)):
            file_path = dataset_path + person + voiceless_wav_path + f
            wav,sr = librosa.load(file_path)
            f0, cp, ap = pw.wav2world(wav.astype(np.float64), sr)
            D = librosa.stft(y=wav,n_fft=1024)#, hop_length = round(sr*0.005))
            sp, phase = librosa.magphase(D)


            if idx == 0:
                cp_all = cp.T
                sp_all = sp
            else:
                cp_all = np.hstack((cp_all,cp.T))
                sp_all = np.hstack((sp_all,sp))

        
            
        y = cp_all.max(axis=1)
        ymax = y.max()
        y /= ymax
        b = np.zeros_like(x)
        b = y/x

        #apply b
        out_path = dataset_path + person+ result_wav_path
        os.makedirs(out_path,exist_ok = True)
        
        for idx, f in enumerate(os.listdir(dataset_path+person+voiced_wav_path)):
            file_path = dataset_path + person + voiced_wav_path + f
            outfile_path = out_path + f

            wav,sr = librosa.load(file_path)
            f0, sp, ap = pw.wav2world(wav.astype(np.float64), sr)

            rsp = sp*b*ymax
            rsp.clip(0,None)


            rf0 = f0
            rap = np.ones_like(ap)#.copy().clip(0.5,None)
            rx = pw.synthesize(rf0, rsp, rap, sr) # synthesize an utterance using the parameters
            D = librosa.stft(y=rx,n_fft=1024)
            sp, phase = librosa.magphase(D)

            if idx == 0:
                rsp_all = sp
            else:
                rsp_all = np.hstack((rsp_all,sp))

        rb = np.mean(sp_all,axis=1)/np.mean(rsp_all,axis=1)

        plt.clf()
        fig = plt.figure()    
        plt.plot(sp_all.max(axis=1))
        plt.plot(rsp_all.max(axis=1))
        plt.plot(rb)
        fig.canvas.draw()
        im = np.array(fig.canvas.renderer.buffer_rgba())
        dst = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        plt.close()

        cv2.imshow('b',dst)
        cv2.waitKey(1)

        for idx, f in enumerate(os.listdir(dataset_path+person+voiced_wav_path)):
            file_path = dataset_path + person + voiced_wav_path + f
            outfile_path = out_path + f

            wav,sr = librosa.load(file_path)
            f0, sp, ap = pw.wav2world(wav.astype(np.float64), sr)

            rsp = sp*b*ymax
            rsp.clip(0,None)


            rf0 = f0
            rap = np.ones_like(ap)#.copy().clip(0.5,None)
            rx = pw.synthesize(rf0, rsp, rap, sr) # synthesize an utterance using the parameters
            D = librosa.stft(y=rx,n_fft=1024)
            sp, phase = librosa.magphase(D)

            rsp = (sp.T*rb).T

            rD = rsp * np.exp(1j*phase)  # 直交形式への変換はlibrosaの関数ないみたいなので、自分で計算する。
            rwav = librosa.istft(rD)

            sf.write(outfile_path, rwav, sr, subtype="PCM_16")


