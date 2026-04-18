import os
import random
from pydub import AudioSegment

dataset = os.listdir("music_dataset")
labels = dataset
print(labels)
instruments_path=[]
import gc # grabage collecter because i don't want this to crash

for label in labels:
    temp= "music_dataset/"+label+"/"
    instruments_path.append(temp) #get all the instrument paths



def makeMixes(num_mixes):
    i = 34046 # i start from the real mixes i made in audactiy
    for num in range(num_mixes): #for loop 
        num_Stems_To_Mix = random.randint(2,3) #mix 2-3 instruments as this shouldn't create many artifacts or issues

        instrument_select= random.sample(instruments_path,num_Stems_To_Mix) # random sample should make it so it doesn't select the same instrument to mix together  

        mix=AudioSegment.silent(3000) # create silence so i can overlay the instruments on top
        for instument in instrument_select:
            stems=[]
            for wav in os.listdir(instument):
                stems.append(instument+wav)
            
            stem = random.choice(stems)#pick a random stem from the instrument folder

            stem_audio = AudioSegment.from_wav(stem)
            stem_audio = stem_audio-10 # remove abour 10 db from the stems just for saftey

            stem_audio.export("audio_separator_dataset/stems/"+str(i)+"_"+str(instument.split("/")[1])+".wav",format="wav")
            mix=mix.overlay(stem_audio)
            del stems # delete stem because it's taking up memory space

        mix=mix.normalize()#nomalize the audio it will make it louder but should make it clean when it's a spectrogram
        mix.export("audio_separator_dataset/mix/"+str(i)+"_mix.wav",format="wav")
        i=i+1
        gc.collect()

makeMixes(3000)