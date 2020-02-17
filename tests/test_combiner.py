import os
import unittest

import numpy as np
import ruamel.yaml
import tensorflow as tf

from model.combiner import Combiner
from preprocessing.preprocessor import Preprocessor


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
        preprocessor = Preprocessor(mel_channels=self.config['mel_channels'],
                                    start_vec_val=self.config['mel_start_vec_value'],
                                    end_vec_val=self.config['mel_end_vec_value'],
                                    tokenizer=combiner.tokenizer)
        train_samples = [preprocessor.encode('repeated text', mel) for mel in test_mels]
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
                                                   pre_dropout=0.5,
                                                   mask_prob=self.config['mask_prob'])
                train_outputs.append(train_output)

        self.assertAlmostEqual(2.6889209747314453, float(train_outputs[-1]['text_mel']['loss']), places=6)
        self.assertAlmostEqual(2.6947453022003174, float(train_outputs[-1]['mel_mel']['loss']), places=6)
        self.assertAlmostEqual(0.0014022162649780512, float(train_outputs[-1]['mel_text']['loss']), places=6)
        self.assertAlmostEqual(0.0011513512581586838, float(train_outputs[-1]['text_text']['loss']), places=6)

        mel_input, text_input = train_samples[0][0], train_samples[0][1]
        pred_mel_text = combiner.mel_text.predict(mel_input, max_length=10)
        pred_text_text = combiner.text_text.predict(text_input, max_length=10)
        pred_text_mel = combiner.text_mel.predict(text_input, max_length=10)
        pred_mel_mel = combiner.mel_mel.predict(mel_input, max_length=10)

        self.assertAlmostEqual(-19.984098434448242, float(tf.reduce_sum(pred_mel_text['logits'])))
        self.assertAlmostEqual(-19.67447280883789, float(tf.reduce_sum(pred_text_text['logits'])))
        self.assertAlmostEqual(-807.0588989257812, float(tf.reduce_sum(pred_text_mel['mel'])))
        self.assertAlmostEqual(-807.8634033203125, float(tf.reduce_sum(pred_mel_mel['mel'])))

        val_outputs = []
        for (batch, (mel, text, stop)) in enumerate(train_dataset):
            val_output = combiner.val_step(text=text,
                                           mel=mel,
                                           stop=stop,
                                           pre_dropout=0.5,
                                           mask_prob=self.config['mask_prob'])
            val_outputs.append(val_output)

        self.assertAlmostEqual(2.0211267471313477, float(val_outputs[-1]['text_mel']['loss']), places=6)
        self.assertAlmostEqual(2.0158276557922363, float(val_outputs[-1]['mel_mel']['loss']), places=6)
        self.assertAlmostEqual(0.0003641040821094066, float(val_outputs[-1]['mel_text']['loss']), places=6)
        self.assertAlmostEqual(0.00036166838253848255, float(val_outputs[-1]['text_text']['loss']), places=6)
