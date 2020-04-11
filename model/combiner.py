import numpy as np
import tensorflow as tf

from model.models import TextMelTransformer
from preprocessing.tokenizer import Tokenizer
from preprocessing.text_processing import _phonemes


class Combiner:
    
    def __init__(self, config: dict):
        self.config = config
        self._check_config()
        self.tokenizer = Tokenizer(list(_phonemes))
        self.learning_rate = np.array(self.config['learning_rate_schedule'])[0, 1].astype(np.float32)
        self.max_r = np.array(self.config['reduction_factor_schedule'])[0, 1].astype(np.int32)
        self.stop_scaling = config.get('stop_loss_scaling', 1.)
    
    def get_model(self):
        return TextMelTransformer(mel_channels=self.config['mel_channels'],
                                  text_model_dimension=self.config['text_model_dimension'],
                                  text_encoder_num_layers=self.config['text_encoder_num_layers'],
                                  text_encoder_num_heads=self.config['text_encoder_num_heads'],
                                  text_encoder_feed_forward_dimension=self.config[
                                      'text_encoder_feed_forward_dimension'],
                                  speech_model_dimension=self.config['speech_model_dimension'],
                                  speech_decoder_prenet_dimension=self.config['speech_decoder_prenet_dimension'],
                                  speech_decoder_num_layers=self.config['speech_decoder_num_layers'],
                                  speech_decoder_num_heads=self.config['speech_decoder_num_heads'],
                                  speech_decoder_feed_forward_dimension=self.config[
                                      'speech_decoder_feed_forward_dimension'],
                                  speech_postnet_conv_filters=self.config['speech_postnet_conv_filters'],
                                  speech_postnet_conv_layers=self.config['speech_postnet_conv_layers'],
                                  speech_postnet_kernel_size=self.config['speech_postnet_kernel_size'],
                                  max_position_encoding=self.config['max_position_encoding'],
                                  dropout_rate=self.config['dropout_rate'],
                                  max_r=self.max_r,
                                  start_vec_value=self.config['mel_start_vec_value'],
                                  end_vec_value=self.config['mel_end_vec_value'],
                                  debug=self.config['debug'], )
    
    def compile_model(self, model):
        model._compile(stop_scaling=self.stop_scaling, optimizer=self.new_adam(self.learning_rate))
    
    @staticmethod
    def new_adam(learning_rate):
        return tf.keras.optimizers.Adam(learning_rate,
                                        beta_1=0.9,
                                        beta_2=0.98,
                                        epsilon=1e-9)
    
    def _check_config(self):
        key_list = ['mel_channels', 'speech_decoder_num_layers', 'text_encoder_num_layers', 'speech_model_dimension',
                    'text_model_dimension', 'speech_decoder_num_heads', 'text_encoder_num_heads',
                    'text_encoder_feed_forward_dimension', 'speech_decoder_feed_forward_dimension',
                    'speech_decoder_prenet_dimension', 'max_position_encoding', 'speech_postnet_conv_filters',
                    'speech_postnet_conv_layers', 'speech_postnet_kernel_size', 'dropout_rate', 'debug',
                    'mel_start_vec_value', 'mel_end_vec_value', ]
        config_keys = set(self.config.keys())
        missing = [key for key in key_list if key not in config_keys]
        assert len(missing) == 0, 'Config is missing the following keys: {}'.format(missing)
