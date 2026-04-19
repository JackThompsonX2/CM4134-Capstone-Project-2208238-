import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import glob
import gc


def makeSpectrogram():
    wav_files = glob.glob(os.path.join('audio_separator_dataset/mix/*.wav'), recursive=True)#load wav files
    i=0      

    for spectro in wav_files:

        freq, sr = librosa.load(spectro) # load into librosa

        mel_spec = librosa.feature.melspectrogram(y=freq, sr=sr, n_fft=2048, hop_length=512, n_mels=128) # covert to mel sepctrogram using STFT

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max) # convert from power to db so it's show the signals

        plt.imsave(os.path.join("Music_mixes_dataset_classification/mix/", f"{i}_mix.png"),log_mel_spec,cmap="gray") # save the spectrograms as greyscale images
        
        plt.close()
        i=i+1
        gc.collect()



makeSpectrogram()