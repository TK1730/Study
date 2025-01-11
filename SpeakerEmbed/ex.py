import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path
import cv2

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 評価関数
criterion_crossentropy = nn.CrossEntropyLoss()
criterion_l1norm = nn.L1Loss()

# 学習結果の保存用ディレクトリ作成
current_time = time.localtime()
file_name = time.strftime("%Y%m%d", current_time)
save_folder = Path("SpeakerEmbed").joinpath(file_name)
save_folder.mkdir(parents=True, exist_ok=True)

# モデル、最適化関数の定義
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
        encoder_p_dropout=encoder_p_dropout,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_kernels=upsample_kernel_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel
    ).to(device)

# ログの初期化と保存
def save_logs(logs, save_folder):
    log_df = pd.DataFrame(logs)
    log_df.to_csv(save_folder.joinpath("log_out.csv"), index=False)

# モデル保存
def save_model(model, model_name, save_folder, loss):
    model_path = save_folder.joinpath(f"{model_name}_bestloss.pth")
    torch.save(model.state_dict(), model_path)
    np.save(save_folder.joinpath(f"{model_name}_bestloss.npy"), np.array([loss]))

# データローダの作成
def create_dataloader(voice_train, voice_test, whisp_train, whisp_test, frame_length, batch_size):
    train_dataset = Clip_dataset(voice_path=voice_train, whisp_path=whisp_train, frame_length=frame_length)
    test_dataset = Clip_dataset(voice_path=voice_test, whisp_path=whisp_test, frame_length=frame_length)
    loader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    }
    return loader

# トレーニング・検証の実行
def train_and_validate(pseud_model, whisp_model, loader, criterion_crossentropy, criterion_l1norm, optim_p, optim_w, 
                       epochs, accumulation_step, save_folder):
    logs = []
    min_valid_loss = float('inf')

    for epoch in range(1, epochs + 1):
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

            running_loss = 0.0

            for j, (v, w) in enumerate(loader[phase]):
                v, w = v.to(device), w.to(device)
                optim_p.zero_grad(set_to_none=True)
                optim_w.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == 'train'):
                    # エンコーダーに入力
                    v_f, v_c = pseud_model(v)
                    w_f, w_c = whisp_model(w)
                    v_f = torch.mean(v_f, dim=(1, 2))
                    w_f = torch.mean(w_f, dim=(1, 2))

                    v_f = (v_f.T / (torch.norm(v_f, dim=1) + eps)).T
                    w_f = (w_f.T / (torch.norm(w_f, dim=1) + eps)).T

                    # 類似度計算
                    logits = torch.matmul(v_f, w_f.T) * t
                    labels = torch.arange(0, v.size(0), dtype=torch.long, device=device)
                    entropy_loss = (criterion_crossentropy(logits, labels) + criterion_crossentropy(logits.T, labels)) / 2
                    loss_p = criterion_l1norm(v_c, v)
                    loss_w = criterion_l1norm(w_c, w)
                    loss = (entropy_loss + loss_p + loss_w) / accumulation_step
                    running_loss += loss.item()

                    if phase == 'train':
                        loss.backward()

                        # 勾配クリッピング
                        nn.utils.clip_grad_value_(pseud_model.parameters(), clip_value=5.0)
                        nn.utils.clip_grad_value_(whisp_model.parameters(), clip_value=5.0)

                        if (j + 1) % accumulation_step == 0:
                            optim_p.step()
                            optim_w.step()

                    else:
                        epoch_val_loss += running_loss

            # ロスの集計
            if phase == 'train':
                epoch_train_loss += running_loss
                embedding_train_loss += entropy_loss.item()
                l1norm_train_loss += loss_p.item() + loss_w.item()
            else:
                embedding_val_loss += entropy_loss.item()
                l1norm_val_loss += loss_p.item() + loss_w.item()

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
        save_logs(logs, save_folder)

        # ベストモデルの保存
        if epoch_val_loss < min_valid_loss:
            save_model(pseud_model, "pseudo_encoder", save_folder, epoch_val_loss)
            save_model(whisp_model, "whisp_encoder", save_folder, epoch_val_loss)
            min_valid_loss = epoch_val_loss

# データロード
voice_train, voice_test = data_load('dataset/jvs_ver2', 'nonpara30w_mean')
whisp_train, whisp_test = data_load('dataset/jvs_ver2', 'whisper10')
loader = create_dataloader(voice_train, voice_test, whisp_train, whisp_test, frame_length, batch_size)

# モデルと最適化関数の準備
pseud_model = create_model()
whisp_model = create_model()
optim_p = Adam(pseud_model.parameters())
optim_w = Adam(whisp_model.parameters())

# トレーニングと検証の実行
train_and_validate(pseud_model, whisp_model, loader, criterion_crossentropy, criterion_l1norm, 
                   optim_p, optim_w, epochs, accumulation_step, save_folder)
