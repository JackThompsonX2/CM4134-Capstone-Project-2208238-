import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import cv2
from PIL import Image
import tensorflow
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

model= keras.models.load_model("CheckPoint.keras")
img = cv2.imread("Music_separation_dataset/train/mix/0_mix.png",0)

img=cv2.resize(img,(128,128))
img= np.reshape(img,(-1, 128, 128, 1))
img=img/255
drum = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0]])

predimg= np.squeeze(model.predict([img,drum]))

prediction= predimg*255
print(prediction.shape)

cv2.imwrite("separated_drum_partcheck.png",prediction)

img = cv2.imread("music_separation_2/train/full_mix/462_full_mix.png",0)

img=cv2.resize(img,(128,128))
img= np.reshape(img,(-1, 128, 128, 1))
img=img/255
acoustic = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0]])

predimg= np.squeeze(model.predict([img,acoustic]))

prediction= predimg*255
print(prediction.shape)

cv2.imwrite("separated_part_acousticcheck.png",prediction)
