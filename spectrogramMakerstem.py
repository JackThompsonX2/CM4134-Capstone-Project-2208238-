import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import glob
import gc



def makeSpectrogram():
    wav_files = glob.glob(os.path.join('audio_separator_dataset/stems/*.wav'), recursive=True)     

    for spectro in wav_files:

        freq, sr = librosa.load(spectro)

        mel_spec = librosa.feature.melspectrogram(y=freq, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(10, 4))

        spectro=spectro.split("\\")[1]
        spectro=spectro.split(".")[0]

        plt.imsave(os.path.join("Music_separation_dataset/stems/", spectro+".png"),log_mel_spec,cmap="gray")
        
        plt.close()
        gc.collect()

makeSpectrogram()