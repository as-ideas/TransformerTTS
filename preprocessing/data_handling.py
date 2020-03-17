import os
from random import Random

import tensorflow as tf


class Dataset:

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
        dataset = tf.data.Dataset.from_generator(lambda: self._datagen(shuffle),
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

    def _datagen(self, shuffle):
        """
        Shuffle once before generating to avoid buffering
        """
        samples = self._samples[:]
        if shuffle:
            # print(f'shuffling files')
            self._random.shuffle(samples)
        return (self.preprocessor(s) for s in samples)


def load_files(metafile,
               meldir,
               num_samples=None):
    samples = []
    count = 0
    alphabet = set()
    with open(metafile, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l_split = l.split('|')
            text = l_split[-1].strip().lower()
            mel_file = os.path.join(str(meldir), l_split[0] + '.npy')
            samples.append((text, mel_file))
            alphabet.update(list(text))
            count += 1
            if num_samples is not None and count > num_samples:
                break
        alphabet = sorted(list(alphabet))
        return samples, alphabet
