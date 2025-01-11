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

from torch.utils.data import DataLoader, random_split
from dataloader.speaker_classfication import Speaker_classfication_dataset
from net.models import PosteriorEncoder1d
from torch.optim import Adam, AdamW
from pathlib import Path

frame_length = 22
in_channels = 80
hidden_channels = 256
inter_channels = 192
encoder_dwn_kernel_size=5
dilation_rate = 2
# enc
kernel_size = ((5, 5), (5, 5), (5, 5))
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

def create_model():
    return PosteriorEncoder1d(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_size=encoder_dwn_kernel_size,
        dilation_rate=dilation_rate,
        n_layers=16,
    )

def create_model_v2():
    return PosteriorEncoder2ds(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
    )

def create_dir():
    """
    学習結果の保存用ディレクトリ作成
    """
    current_time = time.localtime()
    file_name = time.strftime("%Y%m%d", current_time)
    save_folder = Path("Speaker_classfication").joinpath(file_name)
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

def create_dataloader(fpath, ftype, input_type, frame_length, test_rate, batch_size, shuffle=True):
    dataset = Speaker_classfication_dataset(fpath=fpath, ftype=ftype, input_type=input_type, frame_length=frame_length)
    test_size = int(test_rate * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"total: {len(dataset)} train: {len(train_dataset)} test: {len(test_dataset)}")
    
    loader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    }
    return loader


def train_and_validate(model, loader, criterion_crossentropy, optim,
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

            
            for j, (v, label) in enumerate(loader[phase]):
                v, label = v.to(device), label.to(device)
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
    model = create_model().to(device)
    
    loader = create_dataloader('dataset/jvs_ver3', 'whisper10', 'msp', frame_length=frame_length, test_rate=0.2, batch_size=128, shuffle=True)

    # 最適化関数
    optim = AdamW(model.parameters())
    save_folder = create_dir()
    # トレーニングと検証
    train_and_validate(model, loader, criterion_crossentropy, optim, 30000, save_folder)

    