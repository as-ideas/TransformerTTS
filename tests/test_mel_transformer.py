import unittest

import numpy as np
import tensorflow as tf

from losses import masked_mean_squared_error, masked_crossentropy
from model.transformer_factory import new_mel_transformer


class TestMelTransformer(unittest.TestCase):
    
    def setUp(self) -> None:
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def test_training(self):
        start_vec = np.ones((1, 80))
        end_vec = np.ones((1, 80)) * 2
        test_mels = [np.random.random((100 + i * 5, 80)) for i in range(10)]
        train_samples = []
        for mel in test_mels:
            mel = np.concatenate([start_vec, tf.math.log(mel), end_vec])
            stop_probs = np.ones((mel.shape[0]))
            stop_probs[-1] = 2
            train_samples.append((mel, stop_probs))
        
        losses = [masked_mean_squared_error,
                  masked_crossentropy,
                  masked_mean_squared_error]
        loss_coeffs = [1.0, 1.0, 1.0]
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        
        mel_transformer = new_mel_transformer(start_vec=start_vec,
                                              stop_prob_index=2,
                                              num_layers=2,
                                              d_model=64,
                                              num_heads=2,
                                              dff=32,
                                              dff_prenet=32,
                                              max_position_encoding=1000,
                                              dropout_rate=0.1,
                                              mel_channels=80,
                                              postnet_conv_filters=32,
                                              postnet_conv_layers=2,
                                              postnet_kernel_size=5)
        
        mel_transformer.compile(loss=losses, loss_weights=loss_coeffs, optimizer=optimizer)
        train_gen = lambda: (mel for mel in train_samples)
        train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float64, tf.int64))
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.padded_batch(batch_size=2,
                                                   padded_shapes=([-1, 80], [-1]))
        train_dataset = train_dataset.shuffle(10).prefetch(tf.data.experimental.AUTOTUNE)
        
        losses = []
        batch_num = 0
        for epoch in range(2):
            for i, (mels, stop_probs) in enumerate(train_dataset):
                dout = tf.cast(0.5, tf.float32)
                out = mel_transformer.train_step(mels, mels, stop_probs, dout)
                loss = float(out['loss'])
                losses.append(loss)
                print('batch {} loss {}'.format(epoch, loss))
                batch_num += 1
        
        pred = mel_transformer.predict(test_mels[0], max_length=50)
        self.assertAlmostEqual(4.052087783813477, losses[-1], places=6)
        self.assertEqual((50, 80), pred['mel'].numpy().shape)
        self.assertAlmostEqual(-2392.96484375, float(tf.reduce_sum(pred['mel'])), places=6)
