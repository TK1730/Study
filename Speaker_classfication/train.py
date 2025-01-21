import sys
import shutil
import torch
import torch.nn as nn
import torch.functional as F
import cv2
import matplotlib.pyplot as plt
import pickle
import time
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, random_split
from dataloader.speaker_classfication import Speaker_classfication_dataset
from net.models import PosteriorEncoder1d_mish, PosteriorEncoder2ds
from torch.optim import Adam, AdamW
from pathlib import Path
from utils import functions

# setting
epochs = 3000
test_rate = 0.2
batch_size = 64
shuffle = True
learning_rate = 1.0e-3
train_decay = 0.998

dataset_path = 'dataset/jvs_ver3'
ftype = 'nonpara30w_mean'
input_type = 'msp'


in_channels = 80
hidden_channels = 256
encoder_dwn_kernel_size=5
dilation_rate = 2
frame_length = 22
n_layers = 16

p_dropout = 0.1

# enc
kernel_size = ((5, 5), (5, 5), (5, 5))

memo = 'speakerclassfication '

# fc
fc1 = 2048
fc2 = 97

class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, output1, label):
        # コサイン類似性を計算
        cosine_similarity = torch.nn.functional.cosine_similarity(output1, label, dim=1)
        
        # 損失を計算 (1 - コサイン類似性)
        loss = torch.mean((1 - cosine_similarity))
        return loss

# 評価関数
criterion_crossentropy = nn.CrossEntropyLoss()
criterion_cossim = CosineSimilarityLoss()
criterion_mse = nn.MSELoss()

def create_model():
    return PosteriorEncoder1d_mish(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_size=encoder_dwn_kernel_size,
        dilation_rate=dilation_rate,
        frame_length=frame_length,
        n_layers=n_layers,
        p_dropout=p_dropout,
    )

def create_model_v2():
    return PosteriorEncoder2ds(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
    )

def make_folder():
    # 学習結果の保存用ディレクトリ作成
    current_time = time.localtime()
    file_name = time.strftime("%Y%m%d_%H%M", current_time)
    save_folder = Path("Speaker_classfication").joinpath(file_name)
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

# ログの初期化と保存
def save_logs(logs:list, save_folder:Path):
    log_df = pd.DataFrame(logs)
    log_df.to_csv(save_folder.joinpath("log_out.csv"))

# モデル保存
def save_model(model:nn.Module, model_name:str, save_folder:Path):
    model_path = save_folder.joinpath(f'{model_name}_bestloss.pth')
    torch.save(model.state_dict(), model_path)

def save_use_data(data, indices, fpath: Path, fname):
    npsave = []
    for i in indices:
        file_name = data.__getfilename__(i)
        npsave.append(file_name)
    np.savetxt(fpath.joinpath(fname + '.csv'), npsave, delimiter=',', fmt='%s')


def train_and_validate(model, loader, criterion_crossentropy, optim, scheduler,
                       epochs, save_folder):
    logs = []
    min_valid_loss = 10000.0

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch}/{epochs}")
        epoch_train_loss, epoch_val_loss = 0.0, 0.0
        epoch_train_acc, epoch_val_acc = 0.0, 0.0
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            epoch_corrects = 0

            for j, (v, label,) in enumerate(loader[phase]):
                v = v.to(device)
                label = label.to(device).to(torch.float32)

                # optimizer初期化
                optim.zero_grad(set_to_none=True)


                with torch.set_grad_enabled(phase=='train'):
                    # モデルに入力
                    outputs = model(v)
                    loss = criterion_crossentropy(outputs, label)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        # 勾配更新
                        optim.step()

                        # ロスの集計
                        epoch_train_loss += loss.item() / len(loader[phase].dataset)
                    
                    else:
                        epoch_val_loss += loss.item() / len(loader[phase].dataset)
                    
                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == torch.argmax(label, dim=1))
            
            if phase == 'train':
                epoch_train_acc = epoch_corrects.item() / len(loader[phase].dataset)
            else:
                epoch_val_acc = epoch_corrects.item() / len(loader[phase].dataset)

        # スケジューラー 更新
        scheduler.step()

        # ベストモデルの保存
        if epoch_val_loss < min_valid_loss:
            save_model(model, "best_model", save_folder)
            min_valid_loss = epoch_val_loss

        # ログの保存
        log_epoch = {
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'min_val_loss': min_valid_loss,
            'train_acc': epoch_train_acc,
            'val_acc': epoch_val_acc,
        }
        logs.append(log_epoch)
        save_logs(logs, save_folder)
        
        # 表示
        epoch_finish_time = time.time()
        print('timer: {:.4f} sec.'.format(epoch_finish_time - epoch_start_time))
        print(f'epoch:{epoch} train_loss:{epoch_train_loss:.4f} val_loss:{epoch_val_loss:.4f} min_val_loss: {min_valid_loss:.4f}')
        print(f'train_acc: {epoch_train_acc:.4f} val_acc: {epoch_val_acc:.4f}')

if __name__ == '__main__':
    # gpu確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # フォルダ作成
    save_folder = make_folder()
    
    # データロード
    dataset = Speaker_classfication_dataset(fpath=dataset_path, ftype=ftype, input_type=input_type, frame_length=frame_length)
    test_size = int(test_rate * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # データ保存
    save_use_data(dataset, train_dataset.indices,save_folder, 'train_dataset')
    save_use_data(dataset, test_dataset.indices, save_folder, 'test_dataset')
    
    print(f"total: {len(dataset)} train: {len(train_dataset)} test: {len(test_dataset)}")
    
    
    loader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    }
    model = create_model().to(device)
    
    # 最適化関数
    optim = AdamW(model.parameters(), lr=learning_rate)
    # パラメータグループに初期学習率を追加
    for param_group in optim.param_groups: param_group['initial_lr'] = learning_rate
    # スケジューラー設定
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=train_decay, last_epoch=epochs-2)
    # モデルの設定保存
    learning_setting = {
        "train":{
            "epochs": epochs,
            "batch_size": batch_size,
            "test_rate": test_rate,
            "optim": optim.__class__.__name__,
            "schedulur": "ExponentialLR",
            "lr": learning_rate,
            "lr_decacy": train_decay,
        },
        "data":{
            "dataset": dataset_path,
            "ftype": ftype,
            "input_type": input_type,
            "frame_lenght": frame_length,
        },
        "model":{
            "Name": model._get_name(),
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "encoder_dwn_kernel_size": 5,
            "dilation_rate": dilation_rate,
            "frame_length": frame_length,
            "n_layers": n_layers,
            "p_dropout": p_dropout
        },
        "loss": criterion_crossentropy._get_name(),
        "memo": memo,
    }
    functions.Custum(learning_setting, save_folder.joinpath("setting.yml"))
    print("モデルの設定保存完了")
    # トレーニングと検証
    train_and_validate(model, loader, criterion_crossentropy, optim, scheduler, epochs, save_folder)

    