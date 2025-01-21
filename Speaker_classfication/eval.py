import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from dataloader.speaker_classfication import Speaker_classfication_dataset
from net.models import PosteriorEncoder1d
from torch.optim import AdamW
from pathlib import Path

# encoder setting
frame_length = 22
in_channels = 80
hidden_channels = 512
kernel_size=5
dilation_rate = 2

def model_load(model_path):
    model = PosteriorEncoder1d(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=16)
    
    model.load_state_dict(torch.load(model_path))
    return model

if __name__ == '__main__':
    # gpu確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # データセット読み込み
    dataset = Speaker_classfication_dataset(fpath="dataset/jvs_ver3", ftype="whisper10", input_type="msp", frame_length=frame_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    # モデル読み込み
    model = model_load('Speaker_classfication/pseude/best_model_bestloss.pth')
    model.eval()

    for v, label in dataloader:
        v, label = v.to(device), label.to(device)

        with torch.set_grad_enabled():
            print(v.shape)
            # output = model(v)
            # print(v.shape)



