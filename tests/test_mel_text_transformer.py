import unittest

import numpy as np
import tensorflow as tf

from losses import masked_crossentropy
from model.transformer_factory import new_mel_text_transformer


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


class TestMelTextTransformer(unittest.TestCase):
    
    def setUp(self) -> None:
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def test_training(self):
        train_samples = []
        for i in range(10):
            mel = np.random.random((100 + i * 5, 80))
            train_samples.append((mel, 'sample out text.'))
        tokenizer = TestTokenizer(alphabet=sorted(list('sample out text.')))
        start_tok, end_tok = tokenizer.start_token_index, tokenizer.end_token_index
        tokenized_train_samples = [(mel, [start_tok] + tokenizer.encode(text) + [end_tok])
                                   for mel, text in train_samples]
        train_gen = lambda: (pair for pair in tokenized_train_samples)
        train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float64, tf.int64))
        train_dataset = train_dataset.shuffle(10000).padded_batch(4, padded_shapes=([-1, 80], [-1]))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        mel_text_transformer = new_mel_text_transformer(start_token_index=tokenizer.start_token_index,
                                                        end_token_index=tokenizer.end_token_index,
                                                        target_vocab_size=tokenizer.vocab_size,
                                                        num_layers=2,
                                                        d_model=32,
                                                        num_heads=2,
                                                        dff=64,
                                                        dff_prenet=32,
                                                        max_position_encoding=1000,
                                                        dropout_rate=0.1,
                                                        mel_channels=80)
        loss_function = masked_crossentropy
        optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        mel_text_transformer.compile(loss=loss_function, optimizer=optimizer)
        losses = []
        for epoch in range(10):
            for (batch, (inp, tar)) in enumerate(train_dataset):
                output = mel_text_transformer.train_step(inp, tar)
                losses.append(float(output['loss']))
        
        self.assertAlmostEqual(2.841470718383789, losses[-1], places=6)
        pred = mel_text_transformer.predict(tokenized_train_samples[0][0], max_length=10)
        self.assertEqual((1, 1, 19), pred['logits'].numpy().shape)
        self.assertAlmostEqual(-14.454492568969727, float(tf.reduce_sum(pred['logits'])))
