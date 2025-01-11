import sys
sys.path.append('./')
import os
import torch
import time

from torch.utils.data import DataLoader
from dataloader.clip_dataset import Clip_dataset, data_load
frame_length = 128
# CPUのコア数を確認
# os.cpu_count()  # コア数
# print(os.cpu_count())


batch_size = 20

if __name__ == '__main__':
    

    type_list = [0, 2, os.cpu_count()]
    for num in type_list:
        start_time = time.time()
        voice_train, voice_test = data_load(
                'dataset/jvs_ver2',
                'nonpara30w_mean'
            )
        whisp_train, whisp_test = data_load(
            'dataset/jvs_ver2',
            'whisper10'
        )
        # データローダー
        train_dataset = Clip_dataset(voice_path=voice_train,
                                        whisp_path=whisp_train, 
                                        frame_length=frame_length)
        test_dataset = Clip_dataset(voice_path=voice_test,
                                    whisp_path=whisp_train,
                                    frame_length=frame_length,)

        loader = {'train' : DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num),
                    'test' : DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num)}

        for epoch in range(1):
            for phase in ['train', 'test']:
                for j, (v, w) in enumerate(loader[phase]):
                    pass
        print(f'num_worker : {num}')
        print(f'defoult time : {time.time() - start_time}')
