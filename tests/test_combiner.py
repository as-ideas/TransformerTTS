import os
import unittest

import numpy as np
import ruamel.yaml
import tensorflow as tf

from model.combiner import Combiner
from preprocessing.preprocessor import Preprocessor


class TestTokenizer:
    
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet, start=1)}
        self.idx_to_token[0] = '/'
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.start_token_index = len(self.alphabet) + 1
        self.end_token_index = len(self.alphabet) + 2
        self.vocab_size = len(self.alphabet) + 3
        self.idx_to_token[self.start_token_index] = '<'
        self.idx_to_token[self.end_token_index] = '>'
    
    def encode(self, sentence):
        return [self.token_to_idx[c] for c in sentence]
    
    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence])



def encode_text(text, tokenizer):
    encoded_text = tokenizer.encode(text)
    return [tokenizer.start_token_index] + encoded_text + [tokenizer.end_token_index]


class TestCombiner(unittest.TestCase):
    
    def setUp(self) -> None:
        tf.random.set_seed(42)
        np.random.seed(42)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'test_config.yaml')
        yaml = ruamel.yaml.YAML()
        with open(config_path, 'r') as f:
            self.config = yaml.load(f)

    def test_training(self):
        test_mels = [np.random.random((100 + i * 5, 80)) for i in range(10)]
        combiner = Combiner(self.config, tokenizer_alphabet=list('repeated text'))
        preprocessor = Preprocessor(mel_channels=self.config['mel_channels'],
                                    start_vec_val=self.config['mel_start_vec_value'],
                                    end_vec_val=self.config['mel_end_vec_value'],
                                    tokenizer=combiner.tokenizer)
        train_samples = [preprocessor.encode('repeated text', mel)
                         for mel in test_mels]
        train_set_gen = lambda: (item for item in train_samples)
        train_dataset = tf.data.Dataset.from_generator(train_set_gen,
                                                       output_types=(tf.float32, tf.int64, tf.int64))
        train_dataset = train_dataset.shuffle(10).padded_batch(
            self.config['batch_size'], padded_shapes=([-1, 80], [-1], [-1]), drop_remainder=True)

        outputs = []
        for epoch in range(self.config['epochs']):
            for (batch, (mel, text, stop)) in enumerate(train_dataset):
                output = combiner.train_step(text=text,
                                             mel=mel,
                                             stop=stop,
                                             speech_decoder_prenet_dropout=0.5,
                                             mask_prob=self.config['mask_prob'])
                outputs.append(output)

        self.assertAlmostEqual(2.2427563667297363, float(outputs[-1]['text_mel']['loss']), places=6)
        self.assertAlmostEqual(2.268824577331543 , float(outputs[-1]['mel_mel']['loss']), places=6)
        self.assertAlmostEqual(0.0010048405965790153, float(outputs[-1]['mel_text']['loss']), places=6)
        self.assertAlmostEqual(0.0009328527958132327, float(outputs[-1]['text_text']['loss']), places=6)

        mel_input, text_input = train_samples[0][0], train_samples[0][1]
        pred_mel_text = combiner.mel_text.predict(mel_input, max_length=10)
        pred_text_text = combiner.text_text.predict(text_input, max_length=10)
        pred_text_mel = combiner.text_mel.predict(text_input, max_length=10)
        pred_mel_mel = combiner.mel_mel.predict(mel_input, max_length=10)
        self.assertAlmostEqual(-2.70261812210083, float(tf.reduce_sum(pred_mel_text['logits'])))
        self.assertAlmostEqual(-1.2408349514007568, float(tf.reduce_sum(pred_text_text['logits'])))
        self.assertAlmostEqual(-780.1448974609375 , float(tf.reduce_sum(pred_text_mel['mel'])))
        self.assertAlmostEqual(-780.7695922851562, float(tf.reduce_sum(pred_mel_mel['mel'])))

