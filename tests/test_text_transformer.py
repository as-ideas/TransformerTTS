import string
import unittest

import numpy as np
import tensorflow as tf

from src.layers import Encoder, Decoder
from src.models import TextTransformer


class TestTokenizer:

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet)}
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.start_token = len(self.alphabet)
        self.end_token = len(self.alphabet) + 1
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

        train_samples = [('I am a student.', 'Ich bin ein Student.')] * 2
        tokenizer_in = TestTokenizer(alphabet=string.ascii_letters + string.punctuation + string.whitespace)
        tokenizer_out = TestTokenizer(alphabet=string.printable)
        tokenized_train_samples = [(tokenizer_in.encode(i), tokenizer_out.encode(j)) for i, j in train_samples]
        train_gen = lambda: (pair for pair in tokenized_train_samples)
        train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.int64, tf.int64))
        train_dataset = train_dataset.shuffle(10).padded_batch(2, padded_shapes=([-1], [-1]))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        input_vocab_size = tokenizer_in.vocab_size
        target_vocab_size = tokenizer_out.vocab_size

        encoder = Encoder(
            num_layers=2,
            d_model=128,
            num_heads=2,
            dff=256,
            maximum_position_encoding=1000,
            rate=0.1,
        )

        decoder = Decoder(
            num_layers=2,
            d_model=128,
            num_heads=2,
            dff=256,
            maximum_position_encoding=1000,
            rate=0.1,
        )

        transformer = TextTransformer(
            encoder_prenet=tf.keras.layers.Embedding(input_vocab_size, 128),
            decoder_prenet=tf.keras.layers.Embedding(target_vocab_size, 128),
            encoder=encoder,
            decoder=decoder,
            vocab_size={'in': input_vocab_size, 'out': target_vocab_size}
        )

        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        transformer.compile(loss=loss_function, optimizer=optimizer)

        losses = []
        for epoch in range(10):
            for (batch, (inp, tar)) in enumerate(train_dataset):
                gradients, loss, tar_real, predictions = transformer.train_step(inp, tar)
                losses.append(float(loss))

        self.assertAlmostEqual(2.0712099075317383, losses[-1], places=6)
        pred = transformer.predict(tokenized_train_samples[0][0], MAX_LENGTH=10)
        self.assertEqual((1, 1, 102), pred['logits'].numpy().shape)
        self.assertAlmostEqual(-47.13420867919922, float(tf.reduce_sum(pred['logits'])))

