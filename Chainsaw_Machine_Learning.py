import scipy
from scipy import signal
from scipy.io import wavfile

import librosa
import os
import wave
import sys
import numpy as np

from sklearn.mixture import GaussianMixture

directory = 'G:\\Downloads\\ESC-50-master\\ESC-50-master\\audio'
train_X = []
train_y = []

CHUNK = 1024
pre_emphasis = 0.97

def main():
    print("main")
    
    count = 0
    for file in os.listdir(directory):
        filename = file
        count+=1
        print(str(count) + ") " + filename)
                
        filepath = os.path.join(directory, filename)

        sample_rate, samples = wavfile.read(filepath)
        
        target_sample_rate = (16000/sample_rate)*len(samples) # ini rumus dr mana?
        samples = signal.resample(samples, int(target_sample_rate))
        
        # emphasized_signal
        samples = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])
        
        window_size = 16000//2
        for idx in range(0, len(samples), window_size//2):
            window = samples[idx:idx+window_size]
            if len(window) < window_size:
                break

            # mfcc sound
            mfccs = librosa.feature.mfcc(window, sr=sample_rate)

            train_X.append(mfccs)
            if filename.endswith("-41.wav"):
                train_y.append(1)
            else:
                train_y.append(0)
            
        print("train_X length: ", len(train_X))
        
        if (count >= 10):
            break
       
    gmm = GaussianMixture(n_components=4).fit(train_X, train_y)
    print(gmm.shape)

if __name__ == "__main__":
    main()