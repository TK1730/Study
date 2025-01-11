import sys
sys.path.append('./')
import cv2
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import config, functions
from pydub import AudioSegment
from pydub.silence import split_on_silence

def remove_silence(input_path, output_path, silence_thresh=-55):
    """_summary_

    Args:
        input_path (_type_): _description_
        output_path (_type_): _description_
    """
    # 音声ファイル読み込み
    wav = AudioSegment.from_file(input_path)

    # 無音声部分を検出し、音声を分割
    chunks = split_on_silence(
        wav,
        min_silence_len=500, # 500ms以上の無音を検出
        silence_thresh=silence_thresh, # -55db以下を無音と判断
        keep_silence=100, # 無音部分を100ms残す
    )

    # 無音部分を除去した新しい音声を作成
    no_silence_audio = AudioSegment.empty()
    for chunk in chunks:
        no_silence_audio += chunk
    
    # 無音部分を除去した音声を出力
    no_silence_audio.export(output_path, format="wav", bitrate=16)

def distribution(file_path, out_file):
    wav_data = np.load(file_path)
    #RMSを計算
    rms = librosa.feature.rms(y=wav_data, frame_length=config.n_fft, hop_length=config.hop_length) 
    # 音圧レベル (dB) = 20 * log10(RMS振幅 / 基準振幅)
    db=librosa.amplitude_to_db(rms) #dBを計算
    # hist = cv2.calcHist(db, [0], None, 100, range=[-100, -10])
    print(db.max(), db.min())
    hist, bin_edges = np.histogram(db, bins=90, range=[-100, -10])
    print('step1')
    # np.save('data_len.npy', wav_data)
    # db = 20 * np.log(wav_data / p0+1.0e-5)
    # plt.hist(db, bins=10, range=(-100, -10))
    # 表示用に整形する
    hist_df = pd.DataFrame(columns=["start","end","count"])
    for idx, val in enumerate(hist):
        start = round(bin_edges[idx], 2)
        end = round(bin_edges[idx + 1], 2)
        hist_df.loc[idx] = [start, end, val]
    hist_df.to_csv("AudioPreprocessing/" + out_file + '.csv')

def make_wav_stack(folder_path, folder_type, out_path):
    """_summary_
    音声ファイルを読み込み、一次元データとして保存する
    Args:
        folder_path (_type_): 読み込むデータセットのフォルダ
        folder_type (_type_): 読み込むデータのタイプ nonpara30かwhisper10か
        out_path (_type_): 出力するファイル名
    """
    data_list = np.empty(0)
    dataset_path = Path(folder_path)
    for person in dataset_path.iterdir():
        print(person.stem)
        person = person.joinpath(folder_type, "wav24kHz16bit")
        for data in person.iterdir():
            wav, sr = librosa.load(data, sr=config.sr)
            data_list = np.hstack([data_list, wav])
    print(data_list.shape)
    np.save("AudioPreprocessing/" + out_path, data_list)    

def step1():
    # データ一次元化してnpy形式で保存
    data_path = Path("dataset/jvs_ver2")
    make_wav_stack(data_path, "nonpara30w_mean", "nonpara30w_mean_array")
    make_wav_stack(data_path, "whisper10", "whisper10_array")
    # make_wav_stack(data_path, "nonpara30", "nonpara30_array")
    print('done')

def step2():
    folder_path = Path("AudioPreprocessing")
    distribution(folder_path.joinpath("nonpara30w_mean_array.npy"), "nonpara30w_mean_hist")
    distribution(folder_path.joinpath("whisper10_array.npy"), "whisper10_array_hist")
    print("done")

if __name__ == '__main__':
    # step1()
    # step2()    
    dataset_path = Path('dataset/jvs_ver2')
    save_path = Path('dataset/jvs_ver3')
    # whisper10
    for person in dataset_path.iterdir():
        print(person.stem)
        out_path = save_path.joinpath(person.stem, "whisper10", "wav24kHz16bit")
        person = person.joinpath("whisper10", "wav24kHz16bit")
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
        for data in person.iterdir():
            remove_silence(data, out_path.joinpath(data.name), silence_thresh=-81)

    # nonpara30wmean
    for person in dataset_path.iterdir():
        print(person.stem)
        out_path = save_path.joinpath(person.stem, "nonpara30w_mean", "wav24kHz16bit")
        person = person.joinpath("nonpara30w_mean", "wav24kHz16bit")
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
        for data in person.iterdir():
            remove_silence(data, out_path.joinpath(data.name), silence_thresh=-87)
