import librosa
import numpy as np


def invert_griffin_lim(mel, config, n_iter=32):
    mel = np.exp(mel).T
    stft = librosa.feature.inverse.mel_to_stft(mel,
                                               sr=config['sampling_rate'],
                                               n_fft=config['n_fft'],
                                               power=1,
                                               fmin=config['f_min'],
                                               fmax=config['f_max'])
    wav = librosa.feature.inverse.griffinlim(stft,
                                             n_iter=n_iter,
                                             hop_length=config['hop_length'],
                                             win_length=config['win_length'])
    return wav
