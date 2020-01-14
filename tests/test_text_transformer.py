import string
import unittest

import numpy as np
import tensorflow as tf

from losses import masked_crossentropy
from model.transformer_factory import new_text_transformer


class TestTokenizer:
    
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet)}
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.start_token_index = len(self.alphabet)
        self.end_token_index = len(self.alphabet) + 1
        self.vocab_size = len(self.alphabet) + 2
    
    def encode(self, sentence):
        return [self.token_to_idx[c] for c in sentence]
    
    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence])


class TestTextTransformer(unittest.TestCase):
    
    def setUp(self) -> None:
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def test_training(self):
        tokenizer = TestTokenizer(alphabet=string.printable)
        text_transformer = new_text_transformer(tokenizer=tokenizer,
                                                num_layers=2,
                                                d_model=128,
                                                num_heads=2,
                                                dff=256,
                                                max_position_encoding=1000,
                                                dropout_rate=0.1)
        train_samples = [('I am a student.', 'Ich bin ein Student.'), ('I am cool.', 'Ich bin cool.')]
        tokenized_train_samples = [(text_transformer.tokenizer.encode(i), text_transformer.tokenizer.encode(j)) for i, j in train_samples]
        train_gen = lambda: (pair for pair in tokenized_train_samples)
        train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.int64, tf.int64))
        train_dataset = train_dataset.shuffle(10).padded_batch(2, padded_shapes=([-1], [-1]))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        loss_function = masked_crossentropy
        optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        text_transformer.compile(loss=loss_function, optimizer=optimizer)
        
        losses = []
        for epoch in range(10):
            for (batch, (inp, tar)) in enumerate(train_dataset):
                output = text_transformer.train_step(inp, tar)
                losses.append(float(output['loss']))
        
        self.assertAlmostEqual(1.2310866117477417, losses[-1], places=6)
        pred = text_transformer.predict(tokenized_train_samples[0][0], max_length=10)
        self.assertEqual((1, 1, 102), pred['logits'].numpy().shape)
        self.assertAlmostEqual(-22.45075035095215, float(tf.reduce_sum(pred['logits'])))
