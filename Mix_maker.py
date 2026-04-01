import os
import random
from pydub import AudioSegment

dataset = os.listdir("music_dataset")
labels = dataset
print(labels)
instruments_path=[]
import gc

for label in labels:
    temp= "music_dataset/"+label+"/"
    instruments_path.append(temp)



def makeMixes(num_mixes):
    i = 1046
    for num in range(num_mixes):
        num_Stems_To_Mix = random.randint(2,6)
        instrument_select= random.sample(instruments_path,num_Stems_To_Mix)    

        mix=AudioSegment.silent(3000)
        for instument in instrument_select:
            stems=[]
            for wav in os.listdir(instument):
                stems.append(instument+wav)
            
            stem = random.choice(stems)

            stem_audio = AudioSegment.from_wav(stem)

            stem_audio.export("audio_separator_dataset/stems/"+str(i)+"_"+str(instument.split("/")[1])+".wav",format="wav")
            mix=mix.overlay(stem_audio)
            del stems

        mix.export("audio_separator_dataset/mix/"+str(i)+"_mix.wav",format="wav")
        i=i+1
        gc.collect()

makeMixes(2000)