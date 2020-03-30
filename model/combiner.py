import numpy as np
import tensorflow as tf

from utils.decorators import time_it
from utils.losses import masked_mean_squared_error, new_scaled_crossentropy
from model.layers import Encoder, Decoder, SpeechPostnet, SpeechDecoderPrenet
from model.models import TextMelTransformer
from preprocessing.tokenizer import Tokenizer
from preprocessing.text_processing import _phonemes


class Combiner:
    
    def __init__(self, config: dict):
        self.config = config
        self._check_config()
        mel_channels = self.config['mel_channels']
        speech_decoder_num_layers = self.config['speech_decoder_num_layers']
        text_encoder_num_layers = self.config['text_encoder_num_layers']
        speech_model_dimension = self.config['speech_model_dimension']
        text_model_dimension = self.config['text_model_dimension']
        speech_decoder_num_heads = self.config['speech_decoder_num_heads']
        text_encoder_num_heads = self.config['text_encoder_num_heads']
        text_encoder_feed_forward_dimension = self.config['text_encoder_feed_forward_dimension']
        speech_decoder_feed_forward_dimension = self.config['speech_decoder_feed_forward_dimension']
        speech_decoder_prenet_dimension = self.config['speech_decoder_prenet_dimension']
        max_position_encoding = self.config['max_position_encoding']
        speech_postnet_conv_filters = self.config['speech_postnet_conv_filters']
        speech_postnet_conv_layers = self.config['speech_postnet_conv_layers']
        speech_postnet_kernel_size = self.config['speech_postnet_kernel_size']
        dropout_rate = self.config['dropout_rate']
        debug = self.config['debug']
        mel_start_vec_value = self.config['mel_start_vec_value']
        mel_end_vec_value = self.config['mel_end_vec_value']
        self.tokenizer = Tokenizer(list(_phonemes))
        speech_decoder_prenet = SpeechDecoderPrenet(d_model=speech_model_dimension, dff=speech_decoder_prenet_dimension)
        speech_decoder_postnet = SpeechPostnet(mel_channels=mel_channels,
                                               conv_filters=speech_postnet_conv_filters,
                                               conv_layers=speech_postnet_conv_layers,
                                               kernel_size=speech_postnet_kernel_size)
        speech_decoder = Decoder(num_layers=speech_decoder_num_layers,
                                 d_model=speech_model_dimension,
                                 num_heads=speech_decoder_num_heads,
                                 dff=speech_decoder_feed_forward_dimension,
                                 maximum_position_encoding=max_position_encoding,
                                 rate=dropout_rate)
        text_encoder_prenet = tf.keras.layers.Embedding(self.tokenizer.vocab_size, text_model_dimension)
        text_encoder = Encoder(num_layers=text_encoder_num_layers,
                               d_model=text_model_dimension,
                               num_heads=text_encoder_num_heads,
                               dff=text_encoder_feed_forward_dimension,
                               maximum_position_encoding=max_position_encoding,
                               rate=dropout_rate, )
        learning_rate = np.array(self.config['learning_rate_schedule'])[0, 1].astype(np.float32)
        max_r = np.array(self.config['reduction_factor_schedule'])[0, 1].astype(np.int32)
        stop_scaling = config.get('stop_loss_scaling', 1.)
        self.text_mel = TextMelTransformer(encoder_prenet=text_encoder_prenet,
                                           decoder_prenet=speech_decoder_prenet,
                                           decoder_postnet=speech_decoder_postnet,
                                           encoder=text_encoder,
                                           decoder=speech_decoder,
                                           tokenizer=self.tokenizer,
                                           max_r=max_r,
                                           start_vec_value=mel_start_vec_value,
                                           end_vec_value=mel_end_vec_value,
                                           debug=debug)
        self.text_mel.compile(loss=[masked_mean_squared_error,
                                    new_scaled_crossentropy(index=2,
                                                            scaling=stop_scaling),
                                    masked_mean_squared_error],
                              loss_weights=[1., 1., 1.],
                              optimizer=self.new_adam(learning_rate))
    
    @property
    def step(self):
        return int(self.text_mel.optimizer.iterations)
    
    def set_learning_rates(self, new_lr):
        self.text_mel.optimizer.lr.assign(new_lr)
    
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
    
    def train_step(self, text, mel, stop, pre_dropout):
        output = self.text_mel.train_step(text, mel, stop, decoder_prenet_dropout=pre_dropout)
        return output
    
    def val_step(self, text, mel, stop, pre_dropout):
        output = self.text_mel.val_step(text, mel, stop, decoder_prenet_dropout=pre_dropout)
        return output
    
    @time_it
    def predict(self,
                text_seq,
                pre_dropout,
                max_len_mel=1000,
                verbose=True):
        output = self.text_mel.predict(text_seq,
                                       decoder_prenet_dropout=pre_dropout,
                                       max_length=max_len_mel,
                                       verbose=verbose)
        return output
