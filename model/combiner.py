import tensorflow as tf

from losses import masked_crossentropy, masked_mean_squared_error
from model.layers import Encoder, Decoder, SpeechPostnet, PointWiseFFN, SpeechDecoderPrenet, TextPostnet
from model.models import TextTransformer, MelTransformer, MelTextTransformer, TextMelTransformer
from model.transformer_utils import CharTokenizer
from preprocessing.utils import random_text_mask, random_mel_mask


class Combiner:

    def __init__(self, config: dict):
        self.config = config
        self._check_config()
        mel_channels = self.config['mel_channels']
        speech_encoder_num_layers = self.config['speech_encoder_num_layers']
        speech_decoder_num_layers = self.config['speech_decoder_num_layers']
        text_encoder_num_layers = self.config['text_encoder_num_layers']
        text_decoder_num_layers = self.config['text_decoder_num_layers']
        speech_model_dimension = self.config['speech_model_dimension']
        text_model_dimension = self.config['text_model_dimension']
        speech_encoder_num_heads = self.config['speech_encoder_num_heads']
        speech_decoder_num_heads = self.config['speech_decoder_num_heads']
        text_encoder_num_heads = self.config['text_encoder_num_heads']
        text_decoder_num_heads = self.config['text_decoder_num_heads']
        text_encoder_feed_forward_dimension = self.config['text_encoder_feed_forward_dimension']
        text_decoder_feed_forward_dimension = self.config['text_decoder_feed_forward_dimension']
        speech_encoder_feed_forward_dimension = self.config['speech_encoder_feed_forward_dimension']
        speech_decoder_feed_forward_dimension = self.config['speech_decoder_feed_forward_dimension']
        speech_encoder_prenet_dimension = self.config['speech_encoder_prenet_dimension']
        speech_decoder_prenet_dimension = self.config['speech_decoder_prenet_dimension']
        max_position_encoding = self.config['max_position_encoding']
        speech_postnet_conv_filters = self.config['speech_postnet_conv_filters']
        speech_postnet_conv_layers = self.config['speech_postnet_conv_layers']
        speech_postnet_kernel_size = self.config['speech_postnet_kernel_size']
        dropout_rate = self.config['dropout_rate']
        debug = self.config['debug']
        mel_start_vec_value = self.config['mel_start_vec_value']
        mel_end_vec_value = self.config['mel_end_vec_value']
        transformer_kinds = self.config['transformer_kinds']
        alphabet = config['tokenizer_alphabet']
        self.tokenizer = CharTokenizer(list(alphabet))

        speech_encoder_prenet = PointWiseFFN(d_model=speech_model_dimension, dff=speech_encoder_prenet_dimension)
        speech_decoder_prenet = SpeechDecoderPrenet(d_model=speech_model_dimension, dff=speech_decoder_prenet_dimension)
        speech_decoder_postnet = SpeechPostnet(mel_channels=mel_channels,
                                               conv_filters=speech_postnet_conv_filters,
                                               conv_layers=speech_postnet_conv_layers,
                                               kernel_size=speech_postnet_kernel_size)
        speech_encoder = Encoder(num_layers=speech_encoder_num_layers,
                                 d_model=speech_model_dimension,
                                 num_heads=speech_encoder_num_heads,
                                 dff=speech_encoder_feed_forward_dimension,
                                 maximum_position_encoding=max_position_encoding,
                                 rate=dropout_rate)
        speech_decoder = Decoder(num_layers=speech_decoder_num_layers,
                                 d_model=speech_model_dimension,
                                 num_heads=speech_decoder_num_heads,
                                 dff=speech_decoder_feed_forward_dimension,
                                 maximum_position_encoding=max_position_encoding,
                                 rate=dropout_rate)
        text_encoder_prenet = tf.keras.layers.Embedding(self.tokenizer.vocab_size, text_model_dimension)
        text_decoder_prenet = tf.keras.layers.Embedding(self.tokenizer.vocab_size, text_model_dimension)
        text_decoder_postnet = TextPostnet(self.tokenizer.vocab_size)
        text_encoder = Encoder(num_layers=text_encoder_num_layers,
                               d_model=text_model_dimension,
                               num_heads=text_encoder_num_heads,
                               dff=text_encoder_feed_forward_dimension,
                               maximum_position_encoding=max_position_encoding,
                               rate=dropout_rate, )
        text_decoder = Decoder(num_layers=text_decoder_num_layers,
                               d_model=text_model_dimension,
                               num_heads=text_decoder_num_heads,
                               dff=text_decoder_feed_forward_dimension,
                               maximum_position_encoding=max_position_encoding,
                               rate=dropout_rate)
        self.transformer_kinds = transformer_kinds
        self.mel_text, self.text_text, self.text_mel, self.mel_mel = None, None, None, None
        learning_rate = self.config['learning_rate']
        if 'text_mel' in transformer_kinds:
            self.text_mel = TextMelTransformer(encoder_prenet=text_encoder_prenet,
                                               decoder_prenet=speech_decoder_prenet,
                                               decoder_postnet=speech_decoder_postnet,
                                               encoder=text_encoder,
                                               decoder=speech_decoder,
                                               tokenizer=self.tokenizer,
                                               start_vec_value=mel_start_vec_value,
                                               end_vec_value=mel_end_vec_value,
                                               debug=debug)
            self.text_mel.compile(loss=[masked_mean_squared_error,
                                        masked_crossentropy,
                                        masked_mean_squared_error],
                                  loss_weights=[1., 1., 1.],
                                  optimizer=self.new_adam(learning_rate))

        if 'mel_text' in transformer_kinds:
            self.mel_text = MelTextTransformer(encoder_prenet=speech_encoder_prenet,
                                               decoder_prenet=text_decoder_prenet,
                                               decoder_postnet=text_decoder_postnet,
                                               encoder=speech_encoder,
                                               decoder=text_decoder,
                                               tokenizer=self.tokenizer,
                                               mel_channels=mel_channels,
                                               start_vec_value=mel_start_vec_value,
                                               end_vec_value=mel_end_vec_value,
                                               debug=debug)
            self.mel_text.compile(loss=masked_crossentropy,
                                  optimizer=self.new_adam(learning_rate))

        if 'mel_mel' in transformer_kinds:
            self.mel_mel = MelTransformer(encoder_prenet=speech_encoder_prenet,
                                          decoder_prenet=speech_decoder_prenet,
                                          encoder=speech_encoder,
                                          decoder=speech_decoder,
                                          decoder_postnet=speech_decoder_postnet,
                                          start_vec_value=mel_start_vec_value,
                                          end_vec_value=mel_end_vec_value,
                                          debug=debug)
            self.mel_mel.compile(loss=[masked_mean_squared_error,
                                       masked_crossentropy,
                                       masked_mean_squared_error],
                                 loss_weights=[1, 1, 1],
                                 optimizer=self.new_adam(learning_rate))

        if 'text_text' in transformer_kinds:
            self.text_text = TextTransformer(encoder_prenet=text_encoder_prenet,
                                             decoder_prenet=text_decoder_prenet,
                                             decoder_postnet=text_decoder_postnet,
                                             encoder=text_encoder,
                                             decoder=text_decoder,
                                             tokenizer=self.tokenizer,
                                             debug=debug)
            self.text_text.compile(loss=masked_crossentropy,
                                   optimizer=self.new_adam(learning_rate))

    @property
    def step(self):
        return getattr(self, self.transformer_kinds[0]).optimizer.iterations

    def new_adam(self, learning_rate):
        return tf.keras.optimizers.Adam(learning_rate,
                                        beta_1=0.9,
                                        beta_2=0.98,
                                        epsilon=1e-9)

    def _check_config(self):
        key_list = ['mel_channels', 'speech_encoder_num_layers', 'speech_decoder_num_layers',
                    'text_encoder_num_layers', 'text_decoder_num_layers', 'speech_model_dimension',
                    'text_model_dimension', 'speech_encoder_num_heads', 'speech_decoder_num_heads',
                    'text_encoder_num_heads', 'text_decoder_num_heads', 'text_encoder_feed_forward_dimension',
                    'text_decoder_feed_forward_dimension', 'speech_encoder_feed_forward_dimension',
                    'speech_decoder_feed_forward_dimension', 'speech_encoder_prenet_dimension',
                    'speech_decoder_prenet_dimension', 'max_position_encoding', 'speech_postnet_conv_filters',
                    'speech_postnet_conv_layers', 'speech_postnet_kernel_size', 'dropout_rate', 'debug',
                    'mel_start_vec_value', 'mel_end_vec_value', 'transformer_kinds']
        config_keys = set(self.config.keys())
        missing = [key for key in key_list if key not in config_keys]
        assert len(missing) == 0, 'Config is missing the following keys: {}'.format(missing)

    def train_step(self, text, mel, stop, pre_dropout, mask_prob=0.):
        masked_text = random_text_mask(text, mask_prob)
        masked_mel = random_mel_mask(mel, mask_prob)
        output = {}
        if 'mel_mel' in self.transformer_kinds:
            train_out = self.mel_mel.train_step(masked_mel, mel, stop,
                                                decoder_prenet_dropout=pre_dropout)
            output.update({'mel_mel': train_out})
        if 'text_mel' in self.transformer_kinds:
            train_out = self.text_mel.train_step(text, mel, stop,
                                                 decoder_prenet_dropout=pre_dropout)
            output.update({'text_mel': train_out})
        if 'mel_text' in self.transformer_kinds:
            train_out = self.mel_text.train_step(mel, text)
            output.update({'mel_text': train_out})
        if 'text_text' in self.transformer_kinds:
            train_out = self.text_text.train_step(masked_text, text)
            output.update({'text_text': train_out})
        return output

    def predict(self,
                mel,
                text_seq,
                pre_dropout,
                max_len_mel=1000,
                max_len_text=10):
        output = {}
        if 'mel_mel' in self.transformer_kinds and mel is not None:
            pred = self.mel_mel.predict(mel,
                                        decoder_prenet_dropout=pre_dropout,
                                        max_length=max_len_mel)
            output.update({'mel_mel': pred})
        if 'text_mel' in self.transformer_kinds and text_seq is not None:
            pred = self.text_mel.predict(text_seq,
                                         decoder_prenet_dropout=pre_dropout,
                                         max_length=max_len_mel)
            output.update({'text_mel': pred})
        if 'text_text' in self.transformer_kinds and text_seq is not None:
            pred = self.text_text.predict(text_seq,
                                          max_length=max_len_text)
            pred['output_decoded'] = self.tokenizer.decode(pred['output'].numpy())
            output.update({'text_text': pred})
        if 'mel_text' in self.transformer_kinds and mel is not None:
            pred = self.mel_text.predict(mel,
                                         max_length=max_len_text)
            pred['output_decoded'] = self.tokenizer.decode(pred['output'].numpy())
            output.update({'mel_text': pred})
        return output