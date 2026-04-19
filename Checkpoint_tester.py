
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import warnings


warnings.filterwarnings('ignore')



model= keras.models.load_model("CheckPoint.keras")
img = cv2.imread("Music_separation_dataset/test/mix/4044_mix.png",0)

img=img[:128,:128]
img= np.reshape(img,(-1, 128, 128, 1))
img=img/255
drum = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0]])

predimg= np.squeeze(model.predict([img,drum]))

prediction= predimg*255
print(prediction.shape)

cv2.imwrite("separated_drum_partcheck.png",prediction)

img = cv2.imread("Music_separation_dataset/test/mix/4052_mix.png",0)

img=img[:128,:128]
img= np.reshape(img,(-1, 128, 128, 1))
img=img/255
acoustic = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0]])

predimg= np.squeeze(model.predict([img,acoustic]))

prediction= predimg*255
print(prediction.shape)

cv2.imwrite("separated_part_acousticcheck.png",prediction)
