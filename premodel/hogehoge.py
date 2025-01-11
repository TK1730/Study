import sys
sys.path.append('./')
import utils.functions
import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pathlib import Path
def MakeTile(p, d):
    """理想のcos類似度を作る関数 
    Args:
        p (_type_): 話者数
        d (_type_): データ数
    
        image: [1, 1, 0, 0, 0, 0]
               [1, 1, 0, 0, 0, 0]
               [0, 0, 1, 1, 0, 0]
               [0, 0, 1, 1, 0, 0]
               [0, 0, 0, 0, 1, 1]
               [0, 0, 0, 0, 1, 1]
    """
    zeros_tile = np.zeros((p*d, p*d))
    for i in range(p*d):
        if i % d == 0:
            zeros_tile[i:i+d, i:i+d] = 1
    
    return zeros_tile

def MakeSquare(p, d1, d2):
    zeros_tile = np.zeros((p*d1, p*d2))
    for i in range(p):
        zeros_tile[i*d1:i*d1+d1, i*d2:i*d2+d2] = 1
    
    return zeros_tile
person = 10
x = MakeTile(person, 30)
y = MakeTile(person, 10)

z = MakeSquare(person, 30, 10)
print(x.shape)
print(y.shape)
print(z.shape)
x = np.hstack([x, z])
print(x.shape)
y = np.hstack([z.T, y])

cos = np.vstack([x, y])
cv2.imshow('win', cos)
cv2.waitKey(0)
cv2.imwrite("risou.png", cos*255)

        

