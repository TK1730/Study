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
import pickle

from utils import config, functions
from torch.utils.data import DataLoader
from dataloader.clip_dataset import Clip_dataset, data_load
from net.models import VocalNet
from torch.optim import Adam
from pathlib import Path



# setting
epochs = 30000
test_rate = 0.2
batch_size = 10
accumulation_step = 5
frame_length = 64
input_type = ['mcp']
t = 50 # 温度係数
eps = 1e-5

in_channels = 1
hidden_channels = 256
inter_channels = 128
# dec
encoder_dwn_kernel_size = [[5, 4], [5, 4]]
encoder_resblock_kernel_size = [[3, 3], [3, 3], [3, 3],]
encoder_stride_size = [[1, 2], [1, 2]]
encoder_dilation_rate = 2
encoder_resblock_layers = 3
encoder_p_dropout = 0
# deco
resblock_kernel_sizes = [[3], [7], [11]]
resblock_dilation_sizes = [[[1, 1], [3, 1], [5, 1]],
                           [[1, 1], [3, 1], [5, 1]],
                           [[1, 1], [3, 1], [5, 1]]]
upsample_rates = [[1, 2], [1, 2],]
upsample_initial_channel = 512
upsample_kernel_sizes = [[5, 4],[5, 4],]
decoder_LayerNorm = True

# gpu確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 評価関数
criterion_crossentropy = nn.CrossEntropyLoss()
criterion_l1norm = nn.L1Loss()
criterion_mse = nn.MSELoss()

# 学習結果の保存用ディレクトリ作成
current_time = time.localtime()
file_name = time.strftime("%Y%m%d", current_time)
save_folder = Path("SpeakerEmbed").joinpath(file_name)
save_folder.mkdir(parents=True, exist_ok=True)

# モデル, 最適化関数の定義
def create_model():
    return VocalNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            inter_channels=inter_channels,
            encoder_dwn_kernel_size=encoder_dwn_kernel_size,
            encoder_resblock_kernel_size=encoder_resblock_kernel_size,
            encoder_stride_size=encoder_stride_size,
            encoder_dilation_rate=encoder_dilation_rate,
            encoder_resblock_layers=encoder_resblock_layers,
            encoder_LayerNorm=True,
            encoder_p_dropout=encoder_p_dropout,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_kernels=upsample_kernel_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            decoder_LayerNorm=True
            ).to(device)

# ログの初期化と保存
def save_logs(logs:list, save_folder:Path, num):
    log_df = pd.DataFrame(logs)
    log_df.to_csv(save_folder.joinpath("log_out"+str(num)+".csv"))

def save_use_data(data, fpath: Path, fname):
    npsave = [item for sublist in data for item in sublist]
    np.savetxt(fpath.joinpath(fname + '.csv'), npsave, delimiter=',', fmt='%s')
    with open(fpath.joinpath(fname+'.pkl'), 'wb') as f:
        pickle.dump(data, f)

def save_checkpoint(model, optimizer, epoch, fpath, fname='checkpoint.pth'):
    state = {
        'model_1_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, fpath.joinpath(fname))

def load_checkpoint(model, optimizer, fpath, fname='checkpoint.pth'):
    checkpoint = torch.load(fpath.joinpath(fname))
    model.load_state_dict(checkpoint['model_1_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def load_use_data(fpath:Path, fname):
    with open(fpath.joinpath(fname+'.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data

# モデル保存
def save_model(model:nn.Module, model_name:str, save_folder:Path):
    model_path = save_folder.joinpath(f'{model_name}_bestloss.pth')
    torch.save(model.state_dict(), model_path)

def create_dataloader(voice_train, voice_test, whisp_train, whisp_test, frame_length, batch_size):
    train_dataset = Clip_dataset(voice_path=voice_train, whisp_path=whisp_train, frame_length=frame_length)
    test_dataset = Clip_dataset(voice_path=voice_test, whisp_path=whisp_test, frame_length=frame_length)
    loader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    }
    return loader

def cos_sim_show(logits, winname, height=500):
    logits = logits.to('cpu').detach().numpy()
    logits = (logits/t + 1.0) / 2.0
    logits = cv2.resize(logits, dsize=(height, height), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(winname, logits)
    cv2.waitKey(1)

def msp_show(v, w, v_c, w_c, winname):
    v = seikika(v[0].to('cpu').detach().numpy())
    w = seikika(w[0].to('cpu').detach().numpy())
    v_c = seikika(v_c[0].to('cpu').detach().numpy())
    w_c = seikika(w_c[0].to('cpu').detach().numpy())
    img = np.vstack((v.T, v_c.T, w.T, w_c.T))
    img = cv2.resize(img, None, fx=3, fy=3)
    cv2.imshow(winname, img)
    cv2.waitKey(1)

def seikika(x):
    x = (x - x.min()) / (x.max() - x.min()+1.0e-5)
    return x

# トレーニング・検証の実行
def train_and_validate(pseud_model, whisp_model, loader, criterion_crossentropy, criterion_l1norm, optim_p, optim_w,
                       scheduler_p, scheduler_w, epochs, accumulation_step, save_folder):
    logs = []
    min_valid_loss = 100000.0
    num = 7
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

            for j, (v, w) in enumerate(loader[phase]):
                v, w = v.to(device), w.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # エンコーダーに入力
                    v_f, v_c = pseud_model(v)
                    w_f, w_c = whisp_model(w)

                    v_f = torch.mean(v_f, dim=(2, 3))
                    w_f = torch.mean(w_f, dim=(2, 3))
                    vf_l2 = torch.sqrt((v_f**2).sum(dim=1))
                    v_f = (v_f.T/(vf_l2+eps)).T
                    wf_l2 = torch.sqrt((w_f**2).sum(dim=1))
                    w_f = (w_f.T/(wf_l2+eps)).T

                    # 類似度計算
                    logits = torch.matmul(v_f, w_f.T) * t
                    labels = torch.arange(0, v.size(0), dtype=torch.long, device=device)
                    entropy_loss = (criterion_crossentropy(logits, labels) + criterion_crossentropy(logits.T, labels)) / 2
                    loss_p = criterion_l1norm(v_c, v)
                    loss_w = criterion_l1norm(w_c, w)
                    loss = (entropy_loss + loss_p + loss_w) / accumulation_step

                    if phase == 'train':
                        loss.backward()

                        # 勾配クリッピング
                        # nn.utils.clip_grad_value_(pseud_model.parameters(), clip_value=5.0)
                        # nn.utils.clip_grad_value_(whisp_model.parameters(), clip_value=5.0)

                        if (j + 1) % accumulation_step == 0:
                            optim_p.step()
                            optim_w.step()
                            optim_p.zero_grad(set_to_none=True)
                            optim_w.zero_grad(set_to_none=True)
                        
                        # ロスの集計
                        epoch_train_loss += loss.item() * accumulation_step
                        embedding_train_loss += entropy_loss.item() * accumulation_step
                        l1norm_train_loss += (loss_p.item() + loss_w.item()) * accumulation_step

                        if j == 0:
                            # logits
                            cos_sim_show(logits, 'train_logits')
                            # msp
                            msp_show(v, w, v_c, w_c, 'train_msp')
                    else:
                        epoch_val_loss += loss.item() * accumulation_step
                        embedding_val_loss += entropy_loss.item() * accumulation_step
                        l1norm_val_loss += (loss_p.item() + loss_w.item()) * accumulation_step

                        # 画像表示
                        # 類似度
                        if j == 0:
                            # logits
                            cos_sim_show(logits, 'vali_logits')
                            # msp
                            msp_show(v, w, v_c, w_c, 'vali_msp')
                    
        epoch_train_loss = epoch_train_loss / len(loader['train'])
        embedding_train_loss = embedding_train_loss / len(loader['train'])
        l1norm_train_loss = l1norm_train_loss / len(loader['train'])
        epoch_val_loss = epoch_val_loss / len(loader['test'])
        embedding_val_loss = embedding_val_loss / len(loader['test'])
        l1norm_val_loss = l1norm_val_loss / len(loader['test'])

        scheduler_p.step(epoch_val_loss)
        scheduler_w.step(epoch_val_loss)
        # ログ保存
        log_epoch = {
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'embedding_train_loss': embedding_train_loss,
            'embedding_val_loss': embedding_val_loss,
            'l1norm_train_loss': l1norm_train_loss,
            'l1norm_val_loss': l1norm_val_loss
        }
        logs.append(log_epoch)
        # logの保存
        save_logs(logs, save_folder, num)
        # checkpoint
        if epoch % 200 == 0:
            save_checkpoint(model=pseud_model, optimizer=optim_p, epoch=epoch, fpath=save_folder, fname='checkpoint_pseudo_model.pth')
            save_checkpoint(model=whisp_model, optimizer=optim_w, epoch=epoch, fpath=save_folder, fname='checkpoint_whisper_model.pth')
        if (epoch) % 5000 == 0:
            num += 1
            logs = []

        # ベストモデルの保存
        if epoch_val_loss < min_valid_loss:
            save_model(pseud_model, "pseudo_encoder", save_folder)
            save_model(whisp_model, "whisp_encoder", save_folder)
            min_valid_loss = epoch_val_loss

        epoch_finish_time = time.time()
        print('timer: {:.4f} sec.'.format(epoch_finish_time - epoch_start_time))
        print(f'train_loss:{epoch_train_loss:.4f} val_loss:{epoch_val_loss:.4f} min_val_loss: {min_valid_loss:.4f}')

if __name__ == '__main__':
    # データロード
    load = False
    pseud_model = create_model()
    whisp_model = create_model()
    # モデルと最適化関数の準備
    optim_p = Adam(pseud_model.parameters(), lr=0.01)
    optim_w = Adam(whisp_model.parameters(), lr=0.01)
    if not load:
        voice_train, voice_test = data_load('dataset/jvs_ver3', 'nonpara30w_mean')
        whisp_train, whisp_test = data_load('dataset/jvs_ver3', 'whisper10')
        save_use_data(voice_train, save_folder, 'voice_train')
        save_use_data(voice_test, save_folder, 'voice_test')
        save_use_data(whisp_train, save_folder, 'whisp_train')
        save_use_data(whisp_test, save_folder, 'whisp_test')
    else:
        save_folder = Path('SpeakerEmbed/20241111')
        model, _, _ = load_checkpoint(pseud_model, optim_p, save_folder, 'checkpoint_pseudo_model.pth')
        model, _, _ = load_checkpoint(whisp_model, optim_w, save_folder, 'checkpoint_whisper_model.pth')
        voice_train = load_use_data(save_folder, 'voice_train')
        voice_test = load_use_data(save_folder, 'voice_test')
        whisp_train = load_use_data(save_folder, 'whisp_train')
        whisp_test = load_use_data(save_folder, 'whisp_test')

    loader = create_dataloader(voice_train, voice_test, whisp_train, whisp_test, frame_length, batch_size)
    # スケジューラー
    scheduler_p = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_p, mode='min', factor=0.5, patience=1000, min_lr=1e-4, verbose=True)
    scheduler_w = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_w, mode='min', factor=0.5, patience=1000, min_lr=1e-4, verbose=True)

    # トレーニングと検証の実行
    train_and_validate(pseud_model, whisp_model, loader, criterion_crossentropy, criterion_mse,
                       optim_p, optim_w, scheduler_p, scheduler_w, epochs, accumulation_step, save_folder)
