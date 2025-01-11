import itertools
import numpy as np
import random
import torch
import cv2

from torch.utils.data import Dataset
from pathlib import Path

def data_load(fpath:str, ftype:str, input_type:str):
    """データ読み込み

    Args:
        fpath (str): file_path
        ftype (str): wav_type
        input_type (str): input_type

    Returns:
        _type_: _description_
    """
    if type(fpath) != "pathlib.WindowsPath":
        fpath = Path(fpath)
    
    data_list = []
    label_list = []
    for i, person in enumerate(fpath.iterdir()):
        person = person.joinpath(ftype, 'npy')
        for data in person.glob('*' + input_type + '.npy'):
            data_list.append(data)
            label_list.append(i)
    # 0スタートのため+1
    n_labels = i+1 
    for i, label in enumerate(label_list):
        label_list[i] = np.eye(n_labels)[label]
    
    return data_list, label_list


class Speaker_classfication_dataset(Dataset):
    def __init__(self, fpath, ftype, input_type, frame_length):
        # データパスとラベル取得
        self.data_list, self.label_list = data_load(fpath, ftype, input_type)
        # フレーム長さ
        self.frame_length = frame_length

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        voice = self.data_list[idx]
        label = self.label_list[idx]
        # 音声読み込み
        voice = np.load(voice).astype(np.float32)
        # ランダムに切る
        if self.frame_length > 0:
            if self.frame_length >= voice.shape[0]:
                voice = np.pad(voice,[(0,self.frame_length+1-voice.shape[0]),(0,0)], mode='constant')
            st = np.random.randint(0, voice.shape[0]-self.frame_length)
            voice = voice[st:st+self.frame_length].transpose(1, 0)         
                        
        return voice, label

if __name__ == '__main__':
    import sys
    sys.path.append('./')
    # data_load('dataset', 'msp')
    dataset = Speaker_classfication_dataset('dataset/jvs_ver3', 'whisper10', 'msp', 22)
    v, t = dataset.__getitem__(0)
    print(v.shape)
    print(t)
    