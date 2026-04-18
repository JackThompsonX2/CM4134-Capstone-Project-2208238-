import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import glob
import gc


print(os.listdir())
dataset = os.listdir("music_dataset")
labels = dataset
print(labels)

def makeSpectrogram(label):#takes the instrument label
    wav_files = glob.glob(os.path.join('music_dataset/'+label+'/*.wav'), recursive=True)
    i=0
    try:
        os.makedirs("music_dataset_spectro_full_single/train/"+label)
    except:
        print("folder is already made")
    try:
        os.makedirs("music_dataset_spectro_full_single/test/"+label)
    except:
        print("folder is already made")
    try:
        os.makedirs("music_dataset_spectro_full_single/valid/"+label)
    except:
        print("folder is already made")        

    for spectro in wav_files:

        freq, sr = librosa.load(spectro)

        mel_spec = librosa.feature.melspectrogram(y=freq, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max) # convert from power to db so it's show the signals

        if i < int(len(wav_files)*0.80): # have it split the data into training 80%

            plt.imsave(os.path.join("music_dataset_spectro_full_single/train/"+label, f"{i}_"+label+".png"),log_mel_spec,cmap="gray")
        
        elif i >= int(len(wav_files)*0.80) and i < int(len(wav_files)*0.90): # have it split the data into test 10%
            
            plt.imsave(os.path.join("music_dataset_spectro_full_single/test/"+label, f"{i}_"+label+".png"),log_mel_spec,cmap="gray")
        
        else: # have it split the data into valid 10%
            
            plt.imsave(os.path.join("music_dataset_spectro_full_single/valid/"+label, f"{i}_"+label+".png"),log_mel_spec,cmap="gray")
        
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

#makeSpectrogram("Harmonium")
#print("done")

"""""
makeSpectrogram("Saxophone")
print("done")
makeSpectrogram("Shakers")
print("done")
makeSpectrogram("Tambourine")
print("done")
makeSpectrogram("Trombone")
print("done")
makeSpectrogram("Trumpet")
print("done")
makeSpectrogram("Ukulele")
print("done")
makeSpectrogram("vibraphone")
print("done")
makeSpectrogram("Violin")
print("done")

"""""



