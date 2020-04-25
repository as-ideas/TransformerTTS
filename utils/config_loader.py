from typing import Union

import numpy as np
import tensorflow as tf
import ruamel.yaml

from model.models import AutoregressiveTransformer


class ConfigLoader:
    
    def __init__(self, config: Union[dict, str]):
        self.yaml = ruamel.yaml.YAML()
        if isinstance(config, str):
            config = self._load_config(config)
        self.config = config
        self._check_config()
        self.learning_rate = np.array(self.config['learning_rate_schedule'])[0, 1].astype(np.float32)
        self.max_r = np.array(self.config['reduction_factor_schedule'])[0, 1].astype(np.int32)
        self.stop_scaling = config.get('stop_loss_scaling', 1.)
    
    def _load_config(self, config_path):
        return self.yaml.load(open(config_path, 'rb'))
    
    def get_model(self):
        return AutoregressiveTransformer(mel_channels=self.config['mel_channels'],
                                         encoder_model_dimension=self.config['encoder_model_dimension'],
                                         encoder_num_heads=self.config['encoder_num_heads'],
                                         encoder_feed_forward_dimension=self.config[
                                             'encoder_feed_forward_dimension'],
                                         decoder_model_dimension=self.config['decoder_model_dimension'],
                                         decoder_prenet_dimension=self.config['decoder_prenet_dimension'],
                                         decoder_num_heads=self.config['decoder_num_heads'],
                                         decoder_feed_forward_dimension=self.config[
                                             'decoder_feed_forward_dimension'],
                                         postnet_conv_filters=self.config['postnet_conv_filters'],
                                         postnet_conv_layers=self.config['postnet_conv_layers'],
                                         postnet_kernel_size=self.config['postnet_kernel_size'],
                                         max_position_encoding=self.config['max_position_encoding'],
                                         dropout_rate=self.config['dropout_rate'],
                                         max_r=self.max_r,
                                         mel_start_value=self.config['mel_start_value'],
                                         mel_end_value=self.config['mel_end_value'],
                                         phoneme_language=self.config['phoneme_language'],
                                         debug=self.config['debug'])
    
    def compile_model(self, model):
        model._compile(stop_scaling=self.stop_scaling, optimizer=self.new_adam(self.learning_rate))
    
    # TODO: move to model
    @staticmethod
    def new_adam(learning_rate):
        return tf.keras.optimizers.Adam(learning_rate,
                                        beta_1=0.9,
                                        beta_2=0.98,
                                        epsilon=1e-9)
    
    def _check_config(self):
        key_list = ['mel_channels', 'decoder_model_dimension',
                    'encoder_model_dimension', 'decoder_num_heads', 'encoder_num_heads',
                    'encoder_feed_forward_dimension', 'decoder_feed_forward_dimension',
                    'decoder_prenet_dimension', 'max_position_encoding', 'postnet_conv_filters',
                    'postnet_conv_layers', 'postnet_kernel_size', 'dropout_rate', 'debug',
                    'mel_start_value', 'mel_end_value']
        config_keys = set(self.config.keys())
        missing = [key for key in key_list if key not in config_keys]
        assert len(missing) == 0, 'Configuration file error. Missing keys {}'.format(missing)
    
    def dump_config(self, config_path):
        self.yaml.dump(self.config, open(config_path, 'w'))
