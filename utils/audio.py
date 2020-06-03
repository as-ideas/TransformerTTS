import sys

import numpy as np
import librosa


class Audio():
    def __init__(self, config: dict):
        self.config = config
        self.normalizer = getattr(sys.modules[__name__], config['normalizer'])(config)
    
    def _normalize(self, S):
        return self.normalizer.normalize(S)
    
    def _denormalize(self, S):
        return self.normalizer.denormalize(S)
    
    def _linear_to_mel(self, spectrogram):
        return librosa.feature.melspectrogram(
            S=spectrogram,
            sr=self.config['sampling_rate'],
            n_fft=self.config['n_fft'],
            n_mels=self.config['mel_channels'],
            fmin=self.config['f_min'],
            fmax=self.config['f_max'])
    
    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            win_length=self.config['win_length'])
    
    def mel_spectrogram(self, wav):
        """ This is what the model is trained to reproduce. """
        D = self._stft(wav)
        S = self._linear_to_mel(np.abs(D))
        return self._normalize(S)
    
    def reconstruct_waveform(self, mel, n_iter=32):
        """Uses Griffin-Lim phase reconstruction to convert from a normalized
        mel spectrogram back into a waveform."""
        amp_mel = self._denormalize(mel)
        S = librosa.feature.inverse.mel_to_stft(
            amp_mel,
            power=1,
            sr=self.config['sampling_rate'],
            n_fft=self.config['n_fft'],
            fmin=self.config['f_min'],
            fmax=self.config['f_max'])
        wav = librosa.core.griffinlim(
            S,
            n_iter=n_iter,
            hop_length=self.config['hop_length'],
            win_length=self.config['win_length'])
        return wav


class Normalizer:
    def __init__(self, config: dict):
        self.config = config
    
    def normalize(self):
        raise NotImplementedError
    
    def denormalize(self):
        raise NotImplementedError


class MelGAN(Normalizer):
    def __init__(self, config):
        super().__init__(config)
        self.clip_min = 1.0e-5
    
    def normalize(self, S):
        S = np.clip(S, a_min=self.clip_min, a_max=None)
        return np.log(S)
    
    def denormalize(self, S):
        return np.exp(S)


class WaveRNN(Normalizer):
    def __init__(self, config: dict):
        super().__init__(config)
        self.min_level_db = - 100
        self.max_norm = 4
    
    def normalize(self, S):
        S = self.amp_to_db(S)
        S = np.clip((S - self.min_level_db) / -self.min_level_db, 0, 1)
        return (S * 2 * self.max_norm) - self.max_norm
    
    def denormalize(self, S):
        S = (S + self.max_norm) / (2 * self.max_norm)
        S = (np.clip(S, 0, 1) * -self.min_level_db) + self.min_level_db
        return self.db_to_amp(S)
    
    def amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))
    
    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)
