import sys
sys.path.append('./')
import torch
import numpy as np
from net.models import PosteriorEncoder1d

data = torch.load('Speaker_classfication/whisper/best_model_bestloss.pth', map_location=torch.device('cpu'))
print(type(data))
li = ['fc_layers.0.weight', 'fc_layers.0.bias', 'fc_layers.2.weight', 'fc_layers.2.bias']

for i in li:
    del data[i]

torch.save(data, "Speaker_classfication/whisper/pretrain_whisper.pth")
