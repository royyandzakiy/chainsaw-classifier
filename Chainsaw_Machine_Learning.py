import scipy
from scipy import signal
from scipy.io import wavfile

import librosa
import os
import wave
import sys
import numpy as np
import pickle
from sklearn.metrics import classification_report

from sklearn.mixture import GaussianMixture

directory = 'G:\\Downloads\\ESC-50-master\\ESC-50-master\\audio'
train_X = []
train_y = []
test_X = []
test_y = []

CHUNK = 1024
pre_emphasis = 0.97
model_filename = 'gmm_model_2.pkl'

def main():    
    count = 0
    for file in os.listdir(directory):
        filename = file
        if filename.endswith("-41.wav"):
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
                mfccs = librosa.feature.mfcc(window, n_mfcc=39, sr=sample_rate)            
                mfccs_temp = mfccs.mean(axis=0)

                # add training models
        #         if (count <= 50):
                test_X.append(mfccs_temp)            
                test_y.append(0)

            if (count%100 == 0):
                print("count: " + str(count))
                
#             if (count > 30):
#                 break
            
    print("train_X length: ", len(train_X))
    print("test_X length: ", len(test_X))
    print("test_y length: ", len(test_y))
    
    # gmm = GaussianMixture(n_components=4).fit(train_X, train_y)
    
    # save the model to disk
    # pickle.dump(gmm, open(model_filename, 'wb'))
    
    print("DONE!")
    
    # load the model from disk
    loaded_model = pickle.load(open(model_filename, 'rb'))
    result = loaded_model.predict(test_X)
    print("result: ")
    print(classification_report(test_y, result))

if __name__ == "__main__":
    main()
