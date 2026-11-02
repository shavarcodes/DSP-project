import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import librosa
import librosa.display
import IPython.display as ipd
import soundfile as sf
from matplotlib import pyplot as plt
import scipy.fftpack as fft
from scipy.signal import medfilt

audio_file = "C:/Users/shava/OneDrive/Desktop/research audios/tambourine - drum hit - simple.wav"

ipd.Audio(audio_file)
y, sr = librosa.load(audio_file)
print(f'y: {y[:10]}')
print(f'shape y: {y.shape}')
print(f'sr: {sr}')

A = librosa.stft(y)
sound_decibels = librosa.amplitude_to_db(np.abs(A), ref = np.max)
sound_decibels.shape
s, phase = librosa.magphase(A)
power = np.mean(s[:, :int(sr*0.1)], axis = 1)
mask = s > power[:,None]
mask = mask.astype(float)
mask = medfilt(mask, kernel_size=(1,5))
sclean=s * mask
yclean = librosa.istft(sclean * phase)
sf.write('drum hit - simple clean.wav', yclean, sr)



