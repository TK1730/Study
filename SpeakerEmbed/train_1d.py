import sys
sys.path.append('./')
import os
import torch
import torch.nn as nn
import torch.functional as F
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from dataloader.clip_dataset import Clip_dataset, data_load
from net.models import PosteriorEncoder
from torch.optim import Adam
from pathlib import Path

# setting
epochs = 10000
test_rate = 0.2
batch_size = 97
accumulation_step = 256
model_type = 'conv1'
input_type = ['msp']
t = 1.0 # 温度係数
eps = 1e-5

in_channels = 1
hidden_channels = 256

# enc
frame_length = 22
in_channels = 80
hidden_channels = 256
inter_channels = 192
encoder_dwn_kernel_size=5
dilation_rate = 2
n_layers = 16

# gpu確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 評価関数
criterion_crossentropy = nn.CrossEntropyLoss()

# 学習結果の保存用ディレクトリ作成
current_time = time.localtime()
file_name = time.strftime("%Y%m%d", current_time)
save_folder = Path("SpeakerEmbed").joinpath(file_name)
save_folder.mkdir(parents=True, exist_ok=True)

# モデル, 最適化関数の定義
def create_model():
    return PosteriorEncoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_size=encoder_dwn_kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
    ).to(device)

# ログの初期化と保存
def save_logs(logs:list, save_folder:Path):
    log_df = pd.DataFrame(logs)
    log_df.to_csv(save_folder.joinpath("log_out.csv"))

# モデル保存
def save_model(model:nn.Module, model_name:str, save_folder:Path):
    model_path = save_folder.joinpath(f'{model_name}_bestloss.pth')
    torch.save(model.state_dict(), model_path)

def create_dataloader(voice_train, voice_test, whisp_train, whisp_test, frame_length, model_type, batch_size):
    train_dataset = Clip_dataset(voice_path=voice_train, whisp_path=whisp_train, frame_length=frame_length, model_type=model_type)
    test_dataset = Clip_dataset(voice_path=voice_test, whisp_path=whisp_test, frame_length=frame_length, model_type=model_type)
    loader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    }
    return loader

# トレーニング・検証の実行
def train_and_validate(pseud_model, whisp_model, loader, criterion_crossentropy, optim_p, optim_w, epochs, accumulation_step, save_folder):
    logs = []
    min_valid_loss = 100000.0

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch}/{epochs}")
        epoch_train_loss, epoch_val_loss = 0.0, 0.0
        embedding_train_loss, embedding_val_loss = 0.0, 0.0
        l1norm_train_loss, l1norm_val_loss = 0.0, 0.0

        for phase in ['train', 'test']:
            if phase == 'train':
                pseud_model.train()
                whisp_model.train()
            else:
                pseud_model.eval()
                whisp_model.eval()

            for accum in range(accumulation_step):
                for j, (v, w) in enumerate(loader[phase]):
                    v, w = v.to(device), w.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        # エンコーダーに入力
                        v_f = pseud_model(v)
                        w_f = whisp_model(w)
                        v_f = torch.flatten(v_f, start_dim=1)
                        w_f = torch.flatten(w_f, start_dim=1)
                        vf_l2 = torch.sqrt((v_f**2).sum(dim=1))
                        v_f = (v_f.T/(vf_l2+eps)).T
                        wf_l2 = torch.sqrt((w_f**2).sum(dim=1))
                        w_f = (w_f.T/(wf_l2+eps)).T

                        # 類似度計算
                        logits = torch.matmul(v_f, w_f.T) * torch.exp(torch.tensor(t))
                        labels = torch.arange(0, v.size(0), dtype=torch.long, device=device)
                        entropy_loss = (criterion_crossentropy(logits, labels) + criterion_crossentropy(logits.T, labels)) / 2
                        loss = (entropy_loss) / accumulation_step

                        if phase == 'train':
                            loss.backward()

                            if (accum+1) % accumulation_step == 0:
                                optim_p.step()
                                optim_w.step()
                                optim_p.zero_grad(set_to_none=True)
                                optim_w.zero_grad(set_to_none=True)
                            
                            # ロスの集計
                            epoch_train_loss += loss.item() / (accumulation_step)

                        else:
                            epoch_val_loss += loss.item() / (accumulation_step)

                            # 画像表示
                            # 類似度
                            if accum == 0:
                                logits = logits.to('cpu').detach().numpy()
                                logits = (logits/t + 1.0) / 2.0
                                height = 500
                                logits = cv2.resize(logits, dsize=(height, height), interpolation=cv2.INTER_NEAREST)
                                cv2.imshow('output', logits)
                                cv2.waitKey(1)

        # ログ保存
        log_epoch = {
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'min_val_loss': min_valid_loss,
        }
        logs.append(log_epoch)
        save_logs(logs, save_folder)

        # ベストモデルの保存
        if epoch_val_loss < min_valid_loss:
            save_model(pseud_model, "pseudo_encoder", save_folder)
            save_model(whisp_model, "whisp_encoder", save_folder)
            min_valid_loss = epoch_val_loss

        epoch_finish_time = time.time()
        print('timer: {:.4f} sec.'.format(epoch_finish_time - epoch_start_time))
        print(f'epoch:{epoch} train_loss:{epoch_train_loss:.4f} val_loss:{epoch_val_loss:.4f} min_val_loss: {min_valid_loss:.4f}')
        # print(f'v_img: {v_img.max()}, {v_img}')
        # print(f'v_f: {v_f.max()}, {v_f.min()}')
        # print(f'w_f: {w_f.max()}, {w_f.min()}')
        # print(f'v_constency: {v_constency_img.max()}, {v_constency_img.min()}')
        # print(f'w_constency: {w_constency_img.max()}, {w_constency_img.min()}')


if __name__ == '__main__':
    # データロード
    voice_train, voice_test = data_load('dataset/jvs_ver3', 'nonpara30w_mean')
    whisp_train, whisp_test = data_load('dataset/jvs_ver3', 'whisper10')
    print('step1')
    loader = create_dataloader(voice_train, voice_test, whisp_train, whisp_test, frame_length, model_type, batch_size)
    # モデルと最適化関数の準備
    pseud_model = create_model()
    whisp_model = create_model()
    optim_p = Adam(pseud_model.parameters())
    optim_w = Adam(whisp_model.parameters())
    print('step2')
    # モデルロード
    pseud_dic = torch.load('Speaker_classfication/pseude/pretrain_pseude.pth', map_location=device)
    whisp_dic = torch.load('Speaker_classfication/whisper/pretrain_whisper.pth', map_location=device)
    pseud_model.load_state_dict(pseud_dic)
    whisp_model.load_state_dict(whisp_dic)

    # トレーニングと検証の実行
    train_and_validate(pseud_model, whisp_model, loader, criterion_crossentropy, optim_p, optim_w, epochs, accumulation_step, save_folder)
