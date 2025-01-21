import os
import json
import torch
import torch.nn as nn
import torch.functional as F
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import shutil
import sys
import pickle
import net.losses as losses

from torch.utils.data import DataLoader
from dataloader.clip_dataset import Clip_dataset, data_load
from net.models import PosteriorEncoder
from torch.optim import AdamW
from pathlib import Path
from utils import functions

memo = "mish関数のlargeモデルで特徴量を近づける"
# setting
epochs = 10000
batch_size = 97
accumulation_step = 256
test_rate = 0.2
t = 1.0  # 温度係数
eps = 1e-9
learning_rate = 2.0e-4
train_decay = 0.998
seed = 1234

# data
dataset_path = 'dataset/jvs_ver3_small'
voice_type = 'nonpara30w_mean'
whisp_type = 'whisper10'
input_type = 'msp'
frame_length = 22

# model
model_type = 'conv1'
load_model_p = 'Speaker_classfication/pseudo_mish_large/pretrain_pseudo.pth'
load_model_w = 'Speaker_classfication/whisper_mish_large/pretrain_pseudo.pth'

# enc
in_channels = 80
hidden_channels = 256
out_channels = 256
encoder_dwn_kernel_size = 5
dilation_rate = 2
frame_length = 22
n_layers = 16
p_dropout = 0.1
activation = 'mish'

# 評価関数
criterion_crossentropy = nn.CrossEntropyLoss()
criterion_crossentropy._get_name
feature_vec = losses.featire_vec_2

def use_GPU():
    # gpu確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device

def make_folder():
    # 学習結果の保存用ディレクトリ作成
    current_time = time.localtime()
    file_name = time.strftime("%Y%m%d", current_time)
    save_folder = Path("SpeakerEmbed").joinpath(file_name)
    if save_folder.exists():
        print(f"フォルダが存在します")
        user_input = input("フォルダを上書きしますか (y/n): ")
        if user_input == "y":
            shutil.rmtree(save_folder)
            save_folder.mkdir(parents=True, exist_ok=True)
        else:
            print("プログラムを終了します")
            sys.exit()
    else: 
        save_folder.mkdir(parents=True, exist_ok=True)
    return save_folder

# モデル, 最適化関数の定義
def create_model(device):
    return PosteriorEncoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        kernel_size=encoder_dwn_kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        p_dropout=p_dropout,
        activation=activation,
    ).to(device)



# ログの初期化と保存
def save_logs(logs:list, save_folder:Path):
    log_df = pd.DataFrame(logs)
    log_df.to_csv(save_folder.joinpath("log_out.csv"))

def save_use_data(data, fpath: Path, fname):
    npsave = [item for sublist in data for item in sublist]
    np.savetxt(fpath.joinpath(fname + '.csv'), npsave, delimiter=',', fmt='%s')
    with open(fpath.joinpath(fname+'.pkl'), 'wb') as f:
        pickle.dump(data, f)

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

def set_seed(seed): 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 


# トレーニング・検証の実行
def train_and_validate(device, pseud_model, whisp_model, loader, criterion_crossentropy, optim_p, optim_w, epochs, accumulation_step, save_folder):
    logs = []
    min_valid_loss = 100000.0
    # 半精度学習
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch}/{epochs}")
        epoch_train_loss, epoch_val_loss = 0.0, 0.0
        contrast_train_loss, contrast_val_loss = 0.0, 0.0

        for phase in ['train', 'test']:
            if phase == 'train':
                pseud_model.train()
                whisp_model.train()
            else:
                
                pseud_model.eval()
                whisp_model.eval()

            for accum in range(accumulation_step):
                # 勾配の初期化
                optim_p.zero_grad()
                optim_w.zero_grad()
                for j, (v, w) in enumerate(loader[phase]):
                    v, w = v.to(device), w.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        # torch.bfloat16 or torch.float16
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            # z_: 潜在変数z, m_: 平均, v_: 分散
                            v_f = pseud_model(v)
                            w_f = whisp_model(w)

                            # 特徴量
                            v_f = feature_vec(v_f)
                            w_f = feature_vec(w_f)

                            # コントラスティブロス
                            contrast_loss, logits = losses.contrast_loss(v_f, w_f, t=t, eps=eps, device=device)

                            loss = (contrast_loss) / accumulation_step

                        if phase == 'train':
                            scaler.scale(loss).backward()

                            if (accum+1) % accumulation_step == 0:
                                # ampのスケールを戻す
                                scaler.unscale_(optim_p)
                                scaler.unscale_(optim_w)

                                # torch.nn.utils.clip_grad.clip_grad_norm(pseud_model.parameters(), max_norm=20)
                                # torch.nn.utils.clip_grad.clip_grad_norm(whisp_model.parameters(), max_norm=20)

                                # optimizerの更新
                                scaler.step(optim_p)
                                scaler.step(optim_w)
                                scaler.update()
                            
                            # ロスの集計
                            epoch_train_loss += loss.item() / (accumulation_step)
                            contrast_train_loss += contrast_loss.item() / (accumulation_step)

                        else:
                            epoch_val_loss += loss.item() / (accumulation_step)
                            contrast_val_loss += contrast_loss.item() / (accumulation_step)

                            # 画像表示
                            # 類似度
                            if accum == 0:
                                logits = logits.to(torch.float32).to('cpu').detach().numpy()
                                logits = (logits/t + 1.0) / 2.0
                                height = 500
                                logits = cv2.resize(logits, dsize=(height, height), interpolation=cv2.INTER_NEAREST)
                                cv2.imshow('output', logits)
                                cv2.waitKey(1)
                                # 画像の保存
                                if epoch % 100 == 0:
                                    if not save_folder.joinpath('output').exists():
                                        output_folder = save_folder.joinpath('output')
                                        output_folder.mkdir()

                                    cv2.imwrite(f'{output_folder}/logits_{epoch}.png', logits*255)
                                    logits[logits < 0.7] = 0
                                    cv2.imwrite(f'{output_folder}/logits_set{epoch}.png', logits*255)

        

        # ベストモデルの保存
        if epoch_val_loss < min_valid_loss:
            save_model(pseud_model, "pseudo_encoder", save_folder)
            save_model(whisp_model, "whisp_encoder", save_folder)
            min_valid_loss = epoch_val_loss

        # ログ保存
        log_epoch = {
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'min_val_loss': min_valid_loss,
        }
        logs.append(log_epoch)
        save_logs(logs, save_folder)

        epoch_finish_time = time.time()
        print('timer: {:.4f} sec.'.format(epoch_finish_time - epoch_start_time))
        print(f'train_loss:{epoch_train_loss:.6f} val_loss:{epoch_val_loss:.6f} min_val_loss: {min_valid_loss:.6f}')


if __name__ == '__main__':
    # gpu確認
    device = use_GPU()

    # seedの固定
    set_seed(seed)

    # フォルダ作成
    save_folder = make_folder()

    # データロード
    voice_train, voice_test = data_load(dataset_path, voice_type, input_type, test_rate)
    whisp_train, whisp_test = data_load(dataset_path, whisp_type, input_type, test_rate)
    
    # データ保存
    save_use_data(voice_train, save_folder, 'voice_train')
    save_use_data(voice_test, save_folder, 'voice_test')
    save_use_data(whisp_train, save_folder, 'whisp_train')
    save_use_data(whisp_test, save_folder, 'whisp_test')

    # データローダー作成
    loader = create_dataloader(voice_train, voice_test, whisp_train, whisp_test, frame_length, model_type, batch_size)
    print("データロードとデータローダー作成完了")

    # モデルと最適化関数の準備
    pseud_model = create_model(device=device)
    whisp_model = create_model(device=device)
    optim_p = AdamW(pseud_model.parameters(), lr=learning_rate)
    optim_w = AdamW(whisp_model.parameters(), lr=learning_rate)
    print('モデルと最適化関数の準備完了')
    
    # パラメータグループに初期学習率を追加
    for param_group in optim_p.param_groups: param_group['initial_lr'] = learning_rate
    for param_group in optim_w.param_groups: param_group['initial_lr'] = learning_rate

    # スケジューラー設定
    scheduler_p = torch.optim.lr_scheduler.ExponentialLR(optim_p, gamma=train_decay, last_epoch=epochs-2)
    scheduler_w = torch.optim.lr_scheduler.ExponentialLR(optim_w, gamma=train_decay, last_epoch=epochs-2)
    print("スケジューラー設定の完了")

    
    # モデルロード
    pseud_model.load_state_dict(torch.load(load_model_p, map_location=device), strict=False)
    whisp_model.load_state_dict(torch.load(load_model_w, map_location=device), strict=False)
    print("モデルロード完了")

    # モデルの設定保存
    learning_setting = {
        "train":{
            "epochs": epochs,
            "batch_size": batch_size,
            "accmulation_step": accumulation_step,
            "test_rate": test_rate,
            "temperature": t,
            "eps": eps,
            "optim": "Adam",
        },
        "data":{
            "dataset": dataset_path,
            "voice": voice_type,
            "whisp": whisp_type,
            "input_type": input_type,
            "frame_lenght": frame_length,
        },
        "model":{
            "Name": whisp_model._get_name(),
            "load_model_p": load_model_p,
            "load_model_w": load_model_w,
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "encoder_dwn_kernel_size": 5,
            "dilation_rate": dilation_rate,
            "n_layers": n_layers,
            "p_dropout": p_dropout,
            "activation": activation,
        },
        "loss": criterion_crossentropy._get_name(),
        "feature": feature_vec.__name__,
        "optim": optim_p.__class__.__name__,
        "memo": memo,
    }
    functions.Custum(learning_setting, save_folder.joinpath("setting.yml"))
    print("モデルの設定保存完了")

    # トレーニングと検証の実行
    train_and_validate(device, pseud_model, whisp_model, loader, criterion_crossentropy, optim_p, optim_w, epochs, accumulation_step, save_folder)
