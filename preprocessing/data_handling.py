import os
from random import Random
from typing import Union

import tensorflow as tf
import numpy as np


class Tokenizer:
    
    def __init__(self, alphabet, start_token='>', end_token='<', pad_token='/'):
        self.alphabet = alphabet
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet, start=1)}
        self.idx_to_token[0] = pad_token
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.start_token_index = len(self.alphabet) + 1
        self.end_token_index = len(self.alphabet) + 2
        self.vocab_size = len(self.alphabet) + 3
        self.idx_to_token[self.start_token_index] = start_token
        self.idx_to_token[self.end_token_index] = end_token
    
    def encode(self, sentence):
        return [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
    
    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence if int(t) in self.idx_to_token])


class DataPrepper:
    
    def __init__(self,
                 mel_channels: int,
                 start_vec_val: float,
                 end_vec_val: float,
                 tokenizer: Union[Tokenizer],
                 lowercase=True,
                 mel_clip_val=1e-5):
        self.start_vec = np.ones((1, mel_channels)) * start_vec_val
        self.end_vec = np.ones((1, mel_channels)) * end_vec_val
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.mel_channels = mel_channels
        self.mel_clip_val = mel_clip_val
    
    def __call__(self, sample, include_text=True):
        phonemes, text, mel_path = sample
        mel = np.load(mel_path)
        return self._run(phonemes, text, mel, include_text=include_text)
    
    def _run(self, phonemes, text, mel, include_text):
        if self.lowercase:
            phonemes = phonemes.lower()
        encoded_phonemes = self.tokenizer.encode(phonemes)
        encoded_phonemes = [self.tokenizer.start_token_index] + encoded_phonemes + [self.tokenizer.end_token_index]
        norm_mel = np.log(mel.clip(self.mel_clip_val))
        norm_mel = np.concatenate([self.start_vec, norm_mel, self.end_vec], axis=0)
        stop_probs = np.ones((norm_mel.shape[0]))
        stop_probs[-1] = 2
        if include_text:
            return norm_mel, encoded_phonemes, stop_probs, text
        else:
            return norm_mel, encoded_phonemes, stop_probs


class Dataset:
    """ Model digestible dataset. """
    
    def __init__(self,
                 samples,
                 preprocessor,
                 batch_size,
                 shuffle=True,
                 seed=42):
        self._random = Random(seed)
        self._samples = samples[:]
        self.preprocessor = preprocessor
        output_types = (tf.float32, tf.int32, tf.int32)
        padded_shapes = ([-1, preprocessor.mel_channels], [-1], [-1])
        dataset = tf.data.Dataset.from_generator(lambda: self._datagen(shuffle, include_text=False),
                                                 output_types=output_types)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=padded_shapes,
                                       drop_remainder=True)
        self.dataset = dataset
        self.data_iter = iter(dataset.repeat(-1))
    
    def next_batch(self):
        return next(self.data_iter)
    
    def all_batches(self):
        return iter(self.dataset)
    
    def _datagen(self, shuffle, include_text):
        """
        Shuffle once before generating to avoid buffering
        """
        samples = self._samples[:]
        if shuffle:
            # print(f'shuffling files')
            self._random.shuffle(samples)
        return (self.preprocessor(s, include_text) for s in samples)


def load_files(metafile,
               meldir,
               num_samples=None):
    samples = []
    count = 0
    alphabet = set()
    with open(metafile, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l_split = l.split('|')
            mel_file = os.path.join(str(meldir), l_split[0] + '.npy')
            text = l_split[1].strip().lower()
            phonemes = l_split[2].strip()
            samples.append((phonemes, text, mel_file))
            alphabet.update(list(text))
            count += 1
            if num_samples is not None and count > num_samples:
                break
        alphabet = sorted(list(alphabet))
        return samples, alphabet
