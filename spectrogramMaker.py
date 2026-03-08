import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import glob
import gc


print(os.listdir())
dataset = os.listdir("music_dataset")
labels = dataset
print(labels)

def makeSpectrogram(label):
    wav_files = glob.glob(os.path.join('music_dataset/'+label+'/*.wav'), recursive=True)
    i=0
    try:
        os.makedirs("music_dataset_spectro_full/train/"+label)
    except:
        print("folder is already made")
    try:
        os.makedirs("music_dataset_spectro_full/test/"+label)
    except:
        print("folder is already made")
    try:
        os.makedirs("music_dataset_spectro_full/valid/"+label)
    except:
        print("folder is already made")        

    for spectro in wav_files:

        freq, sr = librosa.load(spectro)

        mel_spec = librosa.feature.melspectrogram(y=freq, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
        plt.title('Mel-Spectrogram')
        plt.tight_layout()

        if i < int(len(wav_files)*0.80):

            plt.savefig(os.path.join("music_dataset_spectro_full/train/"+label, f"{i}_spect.png"))
        
        elif i >= int(len(wav_files)*0.80) and i < int(len(wav_files)*0.90):
            
            plt.savefig(os.path.join("music_dataset_spectro_full/test/"+label, f"{i}_spect.png"))
        
        else:
            
            plt.savefig(os.path.join("music_dataset_spectro_full/valid/"+label, f"{i}_spect.png"))
        
        plt.close()
        i=i+1
        gc.collect()


#i would do this in a loop for label in labels but it seems github can commit that many files so i'll do it this way

"""""

makeSpectrogram("Accordion")
print("done")
makeSpectrogram("Acoustic_Guitar")
print("done")
makeSpectrogram("Banjo")
print("done")
makeSpectrogram("Bass_Guitar")
print("done")
makeSpectrogram("Clarinet")
print("done")
makeSpectrogram("cowbell")
print("done")
makeSpectrogram("Cymbals")
print("done")
makeSpectrogram("Dobro")
print("done")
makeSpectrogram("Drum_set")
print("done")
makeSpectrogram("Electric_Guitar")
print("done")
"""""

"""""
makeSpectrogram("Floor_Tom")
print("done")
makeSpectrogram("flute")
print("done")
makeSpectrogram("Harmonica")
print("done")
makeSpectrogram("Hi_Hats")
print("done")
makeSpectrogram("Horn")
print("done")
makeSpectrogram("Keyboard")
print("done")
makeSpectrogram("Mandolin")
print("done")
makeSpectrogram("Organ")
print("done")
makeSpectrogram("Piano")
print("done")
"""""

makeSpectrogram("Harmonium")
print("done")

