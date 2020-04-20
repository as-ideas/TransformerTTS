from typing import Union

import numpy as np

from preprocessing.tokenizer import Tokenizer


class DataPrepper:
    
    def __init__(self,
                 config,
                 tokenizer: Union[Tokenizer]):
        self.start_vec = np.ones((1, config['mel_channels'])) * config['mel_start_value']
        self.end_vec = np.ones((1, config['mel_channels'])) * config['mel_end_value']
        self.tokenizer = tokenizer
        self.mel_channels = config['mel_channels']
    
    def __call__(self, sample, include_text=True):
        phonemes, text, mel_path = sample
        mel = np.load(mel_path)
        return self._run(phonemes, text, mel, include_text=include_text)
    
    def _run(self, phonemes, text, mel, *, include_text):
        encoded_phonemes = self.tokenizer.encode(phonemes)
        norm_mel = np.concatenate([self.start_vec, mel, self.end_vec], axis=0)
        stop_probs = np.ones((norm_mel.shape[0]))
        stop_probs[-1] = 2
        if include_text:
            return norm_mel, encoded_phonemes, stop_probs, text
        else:
            return norm_mel, encoded_phonemes, stop_probs
