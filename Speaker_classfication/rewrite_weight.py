import sys
sys.path.append('./')
import torch
import numpy as np

data = torch.load('Speaker_classfication/whisper_mish_small/best_model_bestloss.pth', map_location=torch.device('cpu'))
print(type(data))
li = ['fc_layers.0.weight', 'fc_layers.0.bias', 'fc_layers.2.weight', 'fc_layers.2.bias']

for i in li:
    del data[i]

torch.save(data, "Speaker_classfication/whisper_mish_small/pretrain_pseudo.pth")
