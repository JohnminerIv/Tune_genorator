import pickle
import numpy as np
from scipy.io.wavfile import write
import librosa
import subprocess
import sys
from keras.models import load_model


def load_data():
    with open('celtic.pickle', 'rb') as f:
        data, labels = pickle.load(f)
    data = data.reshape((10625, 6, 128, 44, 1))
    gen_data = data[:, :5]
    return gen_data


def load():
    return load_model('my_model2.h5')


def predict_from_song_length(len, gen_data, model2):
    beginning = np.random.randint(0, 10624)
    for i in range(5, 5+len):
        if i == 5:
            pred = model2.predict(np.array([gen_data[beginning]]))
        else:
            pred = model2.predict(np.array([gen_spect[i-5:i]]))
        pred = pred.reshape((1, 128, 44, 1))
        if i == 5:
            gen_spect = np.concatenate([gen_data[beginning], pred])
        else:
            gen_spect = np.concatenate([gen_spect, pred])

    gen_spect = gen_spect.reshape((5+len, 128, 44))

    song = []
    for i in range(5+len):
        new_spect = librosa.core.db_to_power(gen_spect[i], ref=1)
        one_second_inter = librosa.feature.inverse.mel_to_audio(new_spect)
        for i in one_second_inter:
            song.append(i)
    song = np.array(song)
    write('output.wav', 22050, song)
    subprocess.call(["afplay", 'output.wav'])


if __name__ == '__main__':
    try:
        your_len = int(sys.argv[1])
    except ValueError:
        print("Oops!  That was no valid number.  Try again...")
    predict_from_song_length(your_len, load_data(), load())
