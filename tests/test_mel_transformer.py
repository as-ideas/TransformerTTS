import unittest

import numpy as np
import tensorflow as tf

from model.layers import Encoder, Decoder, PointWiseFFN, SpeechOutModule
from model.models import MelTransformer


class TestMelTransformer(unittest.TestCase):

    def setUp(self) -> None:
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_training(self):
        start_vec = np.ones((1, 80))
        end_vec = np.ones((1, 80)) * 2
        test_mels = [np.random.random((100 + i*5, 80)) for i in range(10)]
        train_samples = []
        for mel in test_mels:
            mel = np.concatenate([start_vec, mel, end_vec])
            stop_probs = np.zeros((mel.shape[0]))
            stop_probs[-1] = 1
            train_samples.append((mel, stop_probs))

        losses = [tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.MeanAbsoluteError()]
        loss_coeffs = [1.0, 1.0, 1.0]
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        encoder = Encoder(num_layers=2,
                          d_model=64,
                          num_heads=2,
                          dff=32,
                          maximum_position_encoding=1000,
                          rate=0.1)

        decoder = Decoder(num_layers=2,
                          d_model=64,
                          num_heads=2,
                          dff=32,
                          maximum_position_encoding=1000,
                          rate=0.1)

        speech_out_module = SpeechOutModule(mel_channels=80,
                                            conv_filters=32,
                                            conv_layers=2,
                                            kernel_size=5)

        melT = MelTransformer(encoder_prenet=PointWiseFFN(d_model=64, dff=32),
                              decoder_prenet=PointWiseFFN(d_model=64, dff=32),
                              encoder=encoder,
                              decoder=decoder,
                              decoder_postnet=speech_out_module,
                              start_vec=start_vec)

        melT.compile(loss=losses, loss_weights=loss_coeffs, optimizer=optimizer)
        train_gen = lambda: (mel for mel in train_samples)
        train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float64, tf.int64))
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.padded_batch(batch_size=2, padded_shapes=([-1, 80], [-1]))
        train_dataset = train_dataset.shuffle(10).prefetch(tf.data.experimental.AUTOTUNE)

        losses = []
        batch_num = 0
        for epoch in range(2):
            for i, (mels, stop_probs) in enumerate(train_dataset):
                out = melT.train_step(mels, mels, stop_probs)
                loss = float(out['loss'])
                losses.append(loss)
                print('batch {} loss {}'.format(epoch, loss))
                batch_num += 1

        pred = melT.predict(test_mels[0], max_length=50)
        self.assertAlmostEqual(1.2679681777954102, losses[-1], places=6)
        self.assertEqual((50, 80), pred['mel'].numpy().shape)
        self.assertAlmostEqual(2896.39208984375, float(tf.reduce_sum(pred['mel'])), places=6)