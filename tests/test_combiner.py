import os
import unittest

import numpy as np
import ruamel.yaml
import tensorflow as tf

from model.combiner import Combiner
from preprocessing.preprocessor import DataPrepper


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
        combiner = Combiner(self.config)
        data_prep = DataPrepper(mel_channels=self.config['mel_channels'],
                                    start_vec_val=self.config['mel_start_vec_value'],
                                    end_vec_val=self.config['mel_end_vec_value'],
                                    tokenizer=combiner.tokenizer)
        train_samples = [data_prep.encode('repeated text', mel) for mel in test_mels]
        train_set_gen = lambda: (item for item in train_samples)
        train_dataset = tf.data.Dataset.from_generator(train_set_gen,
                                                       output_types=(tf.float32, tf.int32, tf.int32))
        train_dataset = train_dataset.shuffle(10).padded_batch(
            self.config['batch_size'], padded_shapes=([-1, 80], [-1], [-1]), drop_remainder=True)

        train_outputs = []
        for epoch in range(self.config['epochs']):
            for (batch, (mel, text, stop)) in enumerate(train_dataset):
                train_output = combiner.train_step(text=text,
                                                   mel=mel,
                                                   stop=stop,
                                                   pre_dropout=0.5)
                train_outputs.append(train_output)

        self.assertAlmostEqual(2.904730796813965, float(train_outputs[-1]['loss']), places=6)
        mel_input, text_input = train_samples[0][0], train_samples[0][1]
        pred_text_mel = combiner.text_mel.predict(text_input, max_length=10, verbose=False)

        self.assertAlmostEqual(-860.1569213867188, float(tf.reduce_sum(pred_text_mel['mel'])))

        val_outputs = []
        for (batch, (mel, text, stop)) in enumerate(train_dataset):
            val_output = combiner.val_step(text=text,
                                           mel=mel,
                                           stop=stop,
                                           pre_dropout=0.5)
            val_outputs.append(val_output)

        self.assertAlmostEqual(2.307675838470459, float(val_outputs[-1]['loss']), places=6)
