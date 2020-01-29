import os
import unittest

import numpy as np
import ruamel.yaml
import tensorflow as tf

from losses import masked_mean_squared_error, masked_crossentropy
from model.transformer_factory import Combiner


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
        tokenizer = TestTokenizer(alphabet=list('repeated text'))
        test_mels = [np.random.random((100 + i * 5, 80)) for i in range(10)]
        train_samples = []
        combiner = Combiner(self.config, tokenizer.alphabet)
        for i, mel in enumerate(test_mels):
            mel = combiner.transformers['text_to_mel'].preprocess_mel(mel)
            stop_probs = np.ones((mel.shape[0]))
            stop_probs[-1] = 2
            train_samples.append((mel, 'repeated text', stop_probs))

        loss_coeffs = [1.0, 1.0, 1.0]

        mel_text = combiner.transformers['mel_to_text']
        mel_mel = combiner.transformers['mel_to_mel']
        text_mel = combiner.transformers['text_to_mel']
        text_text = combiner.transformers['text_to_text']
        learning_rate = self.config['learning_rate']

        mel_text.compile(loss=masked_crossentropy,
                         optimizer=tf.keras.optimizers.Adam(learning_rate,
                                                            beta_1=0.9,
                                                            beta_2=0.98,
                                                            epsilon=1e-9))
        text_text.compile(loss=masked_crossentropy,
                          optimizer=tf.keras.optimizers.Adam(learning_rate,
                                                             beta_1=0.9,
                                                             beta_2=0.98,
                                                             epsilon=1e-9))
        mel_mel.compile(loss=[masked_mean_squared_error,
                              masked_crossentropy,
                              masked_mean_squared_error],
                        loss_weights=loss_coeffs,
                        optimizer=tf.keras.optimizers.Adam(learning_rate,
                                                           beta_1=0.9,
                                                           beta_2=0.98,
                                                           epsilon=1e-9))
        text_mel.compile(loss=[masked_mean_squared_error,
                               masked_crossentropy,
                               masked_mean_squared_error],
                         loss_weights=loss_coeffs,
                         optimizer=tf.keras.optimizers.Adam(learning_rate,
                                                            beta_1=0.9,
                                                            beta_2=0.98,
                                                            epsilon=1e-9))
        train_tokenized = [(mel, encode_text(text, combiner.tokenizer), stop_prob)
                           for mel, text, stop_prob in train_samples]
        train_set_generator = lambda: (item for item in train_tokenized)
        train_dataset = tf.data.Dataset.from_generator(train_set_generator,
                                                       output_types=(tf.float32, tf.int64, tf.int64))
        train_dataset = train_dataset.shuffle(1000).padded_batch(
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

        self.assertAlmostEqual(2.2208781242370605, float(outputs[-1]['text_to_mel']['loss']), places=6)
        self.assertAlmostEqual(2.245584011077881, float(outputs[-1]['mel_to_mel']['loss']), places=6)
        self.assertAlmostEqual(0.0009718284127302468, float(outputs[-1]['mel_to_text']['loss']), places=6)
        self.assertAlmostEqual(0.000927555956877768, float(outputs[-1]['text_to_text']['loss']), places=6)

        mel_input, text_input = train_tokenized[0][0], train_tokenized[0][1]

        pred_mel_text = mel_text.predict(mel_input, max_length=10)
        pred_text_text = text_text.predict(text_input, max_length=10)
        pred_text_mel = text_mel.predict(text_input, max_length=10)
        pred_mel_mel = mel_mel.predict(mel_input, max_length=10)
        self.assertAlmostEqual(-2.5223498344421387, float(tf.reduce_sum(pred_mel_text['logits'])))
        self.assertAlmostEqual(-1.15986967086792, float(tf.reduce_sum(pred_text_text['logits'])))
        self.assertAlmostEqual(-797.7020263671875, float(tf.reduce_sum(pred_text_mel['mel'])))
        self.assertAlmostEqual(-799.1455078125, float(tf.reduce_sum(pred_mel_mel['mel'])))

