import numpy as np
import librosa


def linear_to_mel(spectrogram, config):
    return librosa.feature.melspectrogram(
        S=spectrogram,
        sr=config['sampling_rate'],
        n_fft=config['n_fft'],
        n_mels=config['mel_channels'],
        fmin=config['f_min'],
        fmax=config['f_max'])


def normalize(S, config):
    S = amp_to_db(S)
    S = np.clip((S - config['min_level_db']) / -config['min_level_db'], 0, 1)
    return (S * 2 * config['max_norm']) - config['max_norm']


def denormalize(S, config):
    S = (S + config['max_norm']) / (2 * config['max_norm'])
    S = (np.clip(S, 0, 1) * -config['min_level_db']) + config['min_level_db']
    return db_to_amp(S)


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def melspectrogram(y, config):
    D = stft(y, config)
    S = linear_to_mel(np.abs(D), config)
    return normalize(S, config)


def stft(y, config):
    return librosa.stft(
        y=y,
        n_fft=config['n_fft'], hop_length=config['hop_length'], win_length=config['win_length'])


def reconstruct_waveform(mel, config, n_iter=32):
    """Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel spectrogram back into a waveform."""
    amp_mel = denormalize(mel, config)
    S = librosa.feature.inverse.mel_to_stft(
        amp_mel,
        power=1,
        sr=config['sampling_rate'],
        n_fft=config['n_fft'],
        fmin=config['f_min'],
        fmax=config['f_max'])
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter,
        hop_length=config['hop_length'], win_length=config['win_length'])
    return wav
