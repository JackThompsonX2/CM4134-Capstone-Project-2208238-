import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import cv2
from PIL import Image


freq, sr = librosa.load("14 I Feel Fine.mp3")

mel_spec = librosa.feature.melspectrogram(y=freq, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

plt.imsave("I_feel_fine.png",log_mel_spec,cmap="gray")


img = cv2.imread("i_feel_fine.png",0)

print(img.shape[1])

patches=[]
for i in range(0,img.shape[1],128):
    patch = img[:,i:i+128]
    print(patch.shape)
    patches.append(patch)
    

combined = patches[0]
for i in range(1,len(patches)):
    try:
        combined = cv2.hconcat([combined, patches[i]])
    except:
        print("end")

print(combined.shape)
cv2.imwrite("recon.png",combined)


