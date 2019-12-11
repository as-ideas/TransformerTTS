import unittest

import numpy as np
import tensorflow as tf

from losses import masked_mean_squared_error, masked_crossentropy
from model.transformer_factory import new_text_mel_transformer


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


class TestTextMelTransformer(unittest.TestCase):
    
    def setUp(self) -> None:
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def test_training(self):
        start_vec = np.ones((1, 80))
        end_vec = np.ones((1, 80)) * 2
        test_mels = [np.random.random((100 + i * 5, 80)) for i in range(10)]
        train_samples = []
        
        for i, mel in enumerate(test_mels):
            mel = np.concatenate([start_vec, mel, end_vec])
            stop_probs = np.ones((mel.shape[0]))
            stop_probs[-1] = 2
            train_samples.append(('repeated text ' * i, mel, stop_probs))
        
        tokenizer = TestTokenizer(alphabet=list('repeated text'))
        
        text_mel_transformer = new_text_mel_transformer(
            start_vec=start_vec,
            stop_prob_index=2,
            input_vocab_size=tokenizer.vocab_size,
            mel_channels=80,
            num_layers=4,
            d_model=256,
            num_heads=4,
            dff=512,
            dff_prenet=256,
            max_position_encoding=1000,
            dropout_rate=0.1,
        )
        
        losses = [masked_mean_squared_error,
                  masked_crossentropy,
                  masked_mean_squared_error]
        loss_coeffs = [1.0, 1.0, 1.0]
        optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        text_mel_transformer.compile(loss=losses, loss_weights=loss_coeffs, optimizer=optimizer)
        start_tok, end_tok = tokenizer.start_token_index, tokenizer.end_token_index
        tokenized_train_samples = [([start_tok] + tokenizer.encode(text) + [end_tok], mel, stops)
                                   for text, mel, stops in train_samples]
        train_gen = lambda: (triple for triple in tokenized_train_samples)
        train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.int64, tf.float64, tf.int64))
        train_dataset = train_dataset.shuffle(10000).padded_batch(2, padded_shapes=([-1], [-1, 80], [-1]))
        train_dataset = train_dataset.shuffle(10).prefetch(tf.data.experimental.AUTOTUNE)
        
        losses = []
        batch_num = 0
        for epoch in range(2):
            for i, (text, mels, stop_probs) in enumerate(train_dataset):
                out = text_mel_transformer.train_step(text, mels, stop_probs)
                loss = float(out['loss'])
                losses.append(loss)
                print('batch {} loss {}'.format(epoch, loss))
                batch_num += 1
        
        pred = text_mel_transformer.predict(tokenized_train_samples[0][0], max_length=50)
        self.assertAlmostEqual(1.6140226125717163, losses[-1], places=6)
        self.assertEqual((50, 80), pred['mel'].numpy().shape)
        self.assertAlmostEqual(2362.845703125, float(tf.reduce_sum(pred['mel'])), places=6)
