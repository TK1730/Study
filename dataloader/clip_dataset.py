import itertools
import numpy as np
import random
import torch
import cv2

from torch.utils.data import Dataset
from pathlib import Path

# データ読み込み
def data_load(dataset, folder, input_type='msp', test_rate=0.2):
    """_summary_

    Args:
        dataset (str): 使用するデータセット
        folder (str): 使用するフォルダーの名前
        np_type (str): 使用する特徴量
    Returns:
        train_data : 学習用データパス
        test_data  : 検証用データパス
    """
    train_data_path = []
    test_data_path = []
    path = Path(dataset)
    if not path.exists():
        print('フォルダーがありません')
    else:
        for person in path.iterdir():
            person = person.joinpath(folder, 'npy')
            # データパス取得
            datas = [str(data) for data in person.iterdir() if input_type in str(data)]
            # データパスをランダムにソート
            random.shuffle(datas)
            train_data, test_data = random_sort(datas, test_rate=test_rate)
            train_data_path.append(train_data)
            test_data_path.append(test_data)

    return train_data_path, test_data_path


def random_sort(data_list:list, test_rate):
    """
    各データリストをtest_rateの割合で分割
    """
    # ランダムなindex番号
    rng = np.random.default_rng()
    data_list = list(rng.permutation(data_list))
    # data 分割
    data_train_length = int(len(data_list) * (1 - test_rate))
    data_train = data_list[:data_train_length]
    data_test = data_list[data_train_length: len(data_list)]
    return data_train, data_test

def Person_sorting(data: list, person=100):
    person_list = ['jvs' + f'{x:03}' for x in range(1, person+1)]
    processing_list = []
    for person in person_list:
        semiprocess_list = []
        for file in data:
            if person in file:
                semiprocess_list.append(file)
        if len(semiprocess_list) != 0:
            processing_list.append(semiprocess_list)
    return processing_list



class Loader(object):
    def __init__(self, voice_array, whisp_array, batch_size, frame_length=128):
        # データ選択用
        self.rng = np.random.default_rng()
        self.voice_person_index = voice_array.shape[0]
        self.whisp_person_index = whisp_array.shape[0]
        self.voice_index = voice_array.shape[1]
        self.whisp_index = whisp_array.shape[1]

        # batch_size
        self.batch_size = batch_size
        # flame_length
        self.frame_length = frame_length

        # 入力された人が合っているか確認
        if self.voice_person_index != self.whisp_person_index:
            print('人数が合いません')
        else:
            print(f"person: {self.voice_person_index}, voice: {self.voice_index}, whisper: {self.whisp_index}")

        self.voice_path = voice_array
        self.whisp_path = whisp_array
        # 1次元配列にする
        self.voice_path = np.ravel(self.voice_path)
        self.whisp_path = np.ravel(self.whisp_path)

        # 1epochによる繰り返し回数
        self.min_data = 1000000
        if self.min_data > self.voice_path.shape[0]: self.min_data = self.voice_path.shape[0]
        if self.min_data > self.whisp_path.shape[0]: self.min_data = self.whisp_path.shape[0]

    def generate_index(self):
        # batchによる繰り返し回数
        iterate = self.voice_person_index * self.voice_index * self.whisp_index // self.batch_size
       
        self.person_index = np.zeros((iterate, self.batch_size), int)
        for i in range(iterate):
            self.person_index[i] = self.rng.choice(self.voice_person_index, size=(1, self.batch_size), replace=False)
        # person_index = self.rng.integers(0, self.voice_person_index, size=(iterate, self.batch_size))    
        v_index = self.rng.integers(0, self.voice_index, size=(iterate, self.batch_size))
        w_index = self.rng.integers(0, self.whisp_index, size=(iterate, self.batch_size))

        # index = 人 * データ + データ
        voice_data_index = self.person_index * self.voice_index + v_index
        whisp_data_index = self.person_index * self.whisp_index + w_index

        pare = np.dstack((voice_data_index, whisp_data_index))
        pare = np.ndarray.transpose(pare, (0, 2, 1))
        return pare

    def ReturnVoice(self, idx):
        d = []
        for i in idx:
            x = np.load(self.voice_path[i]).astype(np.float32)

            if self.frame_length > 0:
                if self.frame_length >= x.shape[0]:
                    x = np.pad(x, [(0, self.frame_length+1-x.shape[0]), (0, 0)], mode='constant')

                st = np.random.randint(0, x.shape[0]-self.frame_length)
                x = x[st:st+self.frame_length]
            d.append(x)
        d = np.array(d)
        d = torch.from_numpy(d).to(torch.float32)
        return d

    def ReturnWhisp(self, idx):
        d = []
        for i in idx:
            x = np.load(self.whisp_path[i]).astype(np.float32)

            if self.frame_length > 0:
                if self.frame_length >= x.shape[0]:
                    x = np.pad(x, [(0, self.frame_length+1-x.shape[0]), (0, 0)], mode='constant')

                st = np.random.randint(0, x.shape[0]-self.frame_length)
                x = x[st:st+self.frame_length]
            d.append(x)
        d = np.array(d)
        d = torch.from_numpy(d).to(torch.float32)
        return d

def data_combination(list_1, list_2):
    """
    二つのリストから各要素の組み合わせを作る
    Args:
        list_1 : voiceのtestかtrainのlist
        list_2 : whisperのtestかtrainのlist
    Return:
        combi : 組み合わせたリスト
    """
    comb_list = []
    for i in range(len(list_1)):
        comb = list(itertools.product(list_1[i], list_2[i], repeat=1))
        comb_list.append(comb)
    comb_list = list(itertools.chain.from_iterable(comb_list))
    return comb_list


class Clip_dataset(Dataset):
    def __init__(self, voice_path, whisp_path, frame_length=128, model_type='conv1'):
        # データ選択用
        self.voice_path = voice_path
        self.whisp_path = whisp_path
        self.frame_length = frame_length
        self.model_type = model_type
    
    def __len__(self):
        return len(self.voice_path)

    def __getitem__(self, idx):
        """
        idx : 人のid番号
        """
        # file index取得
        vsp_file_index = np.random.randint(0, len(self.voice_path[idx]))
        wsp_file_index = np.random.randint(0, len(self.whisp_path[idx]))
        vsp = np.load(self.voice_path[idx][vsp_file_index]).astype(np.float32)
        wsp = np.load(self.whisp_path[idx][wsp_file_index]).astype(np.float32)

        if self.frame_length > 0:
            if self.frame_length >= vsp.shape[0]:
                vsp = np.pad(vsp,[(0,self.frame_length+1-vsp.shape[0]),(0,0)], mode='constant')
            st = np.random.randint(0, vsp.shape[0]-self.frame_length)
            vsp = vsp[st:st+self.frame_length]
                
            if self.frame_length >= wsp.shape[0]:
                wsp = np.pad(wsp,[(0,self.frame_length+1-wsp.shape[0]),(0,0)], mode='constant')
            st = np.random.randint(0, wsp.shape[0]-self.frame_length)
            wsp = wsp[st:st+self.frame_length]
            if self.model_type == 'conv1':
                vsp = vsp.transpose(1, 0)
                wsp = wsp.transpose(1, 0)
            else:
                vsp = vsp.squeeze(0)
                wsp = vsp.squeeze(0)

        return vsp, wsp





if __name__ == '__main__':
    voice_train, voice_test = data_load('dataset/clip_dataset', 'nonpara30')
    whisp_train, whisp_test = data_load('dataset/clip_dataset', 'whisper10')
    print(len(voice_train))
