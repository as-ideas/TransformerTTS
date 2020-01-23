import string

import tensorflow as tf

from model.layers import Encoder, Decoder, SpeechPostnet, PointWiseFFN, SpeechDecoderPrenet, TextPostnet
from model.models import TextTransformer, MelTransformer, MelTextTransformer, TextMelTransformer
from model.transformer_utils import CharTokenizer


class Combiner:  # (tf.keras.Model):
    def __init__(self,
                 *,
                 tokenizer_type: str,
                 mel_channels: int,
                 speech_encoder_num_layers: int,
                 speech_decoder_num_layers: int,
                 text_encoder_num_layers: int,
                 text_decoder_num_layers: int,
                 speech_model_dimension: int,
                 text_model_dimension: int,
                 speech_encoder_num_heads: int,
                 speech_decoder_num_heads: int,
                 text_encoder_num_heads: int,
                 text_decoder_num_heads: int,
                 text_encoder_feed_forward_dimension: int,
                 text_decoder_feed_forward_dimension: int,
                 speech_encoder_feed_forward_dimension: int,
                 speech_decoder_feed_forward_dimension: int,
                 speech_encoder_prenet_dimension: int,
                 speech_decoder_prenet_dimension: int,
                 max_position_encoding: int,
                 speech_postnet_conv_filters: int,
                 speech_postnet_conv_layers: int,
                 speech_postnet_kernel_size: int,
                 dropout_rate: float,
                 mel_start_vec_value: int,
                 mel_end_vec_value: int,
                 tokenizer_alphabet: list = None,
                 debug: bool=False,
                 transformer_kinds=None):
        
        # super(Combiner, self).__init__()

        if transformer_kinds is None:
            transformer_kinds = ['text_to_text', 'mel_to_mel', 'text_to_mel', 'mel_to_text']
        if tokenizer_type == 'char':
            if tokenizer_alphabet is None:
                tokenizer_alphabet = string.printable
            self.tokenizer = CharTokenizer(alphabet=sorted(list(tokenizer_alphabet)))
        else:
            raise NotImplementedError(f'{tokenizer_type} tokenizer not implemented.')
        
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
        self.transformers = {'text_to_mel': TextMelTransformer(encoder_prenet=text_encoder_prenet,
                                                               decoder_prenet=speech_decoder_prenet,
                                                               decoder_postnet=speech_decoder_postnet,
                                                               encoder=text_encoder,
                                                               decoder=speech_decoder,
                                                               tokenizer=self.tokenizer,
                                                               start_vec_value=mel_start_vec_value,
                                                               end_vec_value=mel_end_vec_value,
                                                               debug=debug),
                             'mel_to_text': MelTextTransformer(encoder_prenet=speech_encoder_prenet,
                                                               decoder_prenet=text_decoder_prenet,
                                                               decoder_postnet=text_decoder_postnet,
                                                               encoder=speech_encoder,
                                                               decoder=text_decoder,
                                                               tokenizer=self.tokenizer,
                                                               mel_channels=mel_channels,
                                                               start_vec_value=mel_start_vec_value,
                                                               end_vec_value=mel_end_vec_value,
                                                               debug=debug),
                             'mel_to_mel': MelTransformer(encoder_prenet=speech_encoder_prenet,
                                                          decoder_prenet=speech_decoder_prenet,
                                                          encoder=speech_encoder,
                                                          decoder=speech_decoder,
                                                          decoder_postnet=speech_decoder_postnet,
                                                          start_vec_value=mel_start_vec_value,
                                                          end_vec_value=mel_end_vec_value,
                                                          debug=debug),
                             'text_to_text': TextTransformer(encoder_prenet=text_encoder_prenet,
                                                             decoder_prenet=text_decoder_prenet,
                                                             decoder_postnet=text_decoder_postnet,
                                                             encoder=text_encoder,
                                                             decoder=text_decoder,
                                                             tokenizer=self.tokenizer,
                                                             debug=debug)}
    
    @staticmethod
    def random_mel_mask(tensor, mask_prob):
        tensor_shape = tf.shape(tensor)
        mask_floats = tf.random.uniform((tensor_shape[0], tensor_shape[1]))
        mask = tf.cast(mask_floats > mask_prob, tf.float32)
        mask = tf.expand_dims(mask, -1)
        mask = tf.broadcast_to(mask, tensor_shape)
        masked_tensor = tensor * mask
        return masked_tensor
    
    @staticmethod
    def random_text_mask(tensor, mask_prob):
        tensor_shape = tf.shape(tensor)
        mask_floats = tf.random.uniform((tensor_shape[0], tensor_shape[1]))
        mask = tf.cast(mask_floats > mask_prob, tf.int64)
        masked_tensor = tensor * mask
        return masked_tensor
    
    def train_step(self, text, mel, stop, speech_decoder_prenet_dropout, mask_prob=0.):
        masked_text = self.random_text_mask(text, mask_prob)
        masked_mel = self.random_mel_mask(mel, mask_prob)
        output = {}
        if 'mel_to_mel' in self.transformer_kinds:
            output.update({'mel_to_mel': self.transformers['mel_to_mel'].train_step(masked_mel, mel, stop,
                                                                     decoder_prenet_dropout=speech_decoder_prenet_dropout)})
        if 'text_to_mel' in self.transformer_kinds:
            output.update({'text_to_mel': self.transformers['text_to_mel'].train_step(text, mel, stop,
                                                                       decoder_prenet_dropout=speech_decoder_prenet_dropout)})
        if 'mel_to_text' in self.transformer_kinds:
            output.update({'mel_to_text': self.transformers['mel_to_text'].train_step(mel, text)})
        if 'text_to_text' in self.transformer_kinds:
            output.update({'text_to_text': self.transformers['text_to_text'].train_step(masked_text, text)})
        
        return output
    
    def save_weights(self, path, steps):
        for kind in self.transformer_kinds:
            self.transformers[kind].save_weights(f'{path}/{kind}_weights_steps{steps}.hdf5')


def new_text_transformer(tokenizer,
                         num_layers=1,
                         d_model=64,
                         num_heads=1,
                         dff=512,
                         max_position_encoding=10000,
                         dropout_rate=0,
                         debug=False):
    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=max_position_encoding,
        rate=dropout_rate,
    )
    
    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=max_position_encoding,
        rate=dropout_rate,
    )
    
    text_transformer = TextTransformer(
        encoder_prenet=tf.keras.layers.Embedding(tokenizer.vocab_size, d_model),
        decoder_prenet=tf.keras.layers.Embedding(tokenizer.vocab_size, d_model),
        decoder_postnet=TextPostnet(tokenizer.vocab_size),
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        debug=debug
    )
    
    return text_transformer


def new_mel_transformer(num_layers=1,
                        d_model=64,
                        num_heads=1,
                        dff=512,
                        dff_prenet=512,
                        max_position_encoding=10000,
                        dropout_rate=0,
                        mel_channels=80,
                        postnet_conv_filters=32,
                        postnet_conv_layers=2,
                        postnet_kernel_size=5,
                        start_vec_value=-3,
                        end_vec_value=1,
                        debug=False):
    encoder = Encoder(num_layers=num_layers,
                      d_model=d_model,
                      num_heads=num_heads,
                      dff=dff,
                      maximum_position_encoding=max_position_encoding,
                      rate=dropout_rate)
    
    decoder = Decoder(num_layers=num_layers,
                      d_model=d_model,
                      num_heads=num_heads,
                      dff=dff,
                      maximum_position_encoding=max_position_encoding,
                      rate=dropout_rate)
    
    speech_out_module = SpeechPostnet(mel_channels=mel_channels,
                                      conv_filters=postnet_conv_filters,
                                      conv_layers=postnet_conv_layers,
                                      kernel_size=postnet_kernel_size)
    
    mel_transformer = MelTransformer(encoder_prenet=PointWiseFFN(d_model=d_model, dff=dff_prenet),
                                     decoder_prenet=SpeechDecoderPrenet(d_model=d_model, dff=dff_prenet),
                                     encoder=encoder,
                                     decoder=decoder,
                                     decoder_postnet=speech_out_module,
                                     start_vec_value=start_vec_value,
                                     end_vec_value=end_vec_value,
                                     debug=debug)
    
    return mel_transformer


def new_mel_text_transformer(tokenizer,
                             mel_channels=80,
                             num_layers=1,
                             d_model=64,
                             num_heads=1,
                             dff=512,
                             dff_prenet=512,
                             max_position_encoding=10000,
                             dropout_rate=0,
                             start_vec_value=-3,
                             end_vec_value=1,
                             debug=False):
    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=max_position_encoding,
        rate=dropout_rate,
    )
    
    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=max_position_encoding,
        rate=dropout_rate,
    )
    
    mel_text_transformer = MelTextTransformer(
        encoder_prenet=PointWiseFFN(d_model=d_model, dff=dff_prenet),
        decoder_prenet=tf.keras.layers.Embedding(tokenizer.vocab_size, d_model),
        decoder_postnet=TextPostnet(tokenizer.vocab_size),
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        mel_channels=mel_channels,
        start_vec_value=start_vec_value,
        end_vec_value=end_vec_value,
        debug=debug)
    
    return mel_text_transformer


def new_text_mel_transformer(tokenizer,
                             mel_channels=80,
                             num_layers=1,
                             d_model=64,
                             num_heads=1,
                             dff=512,
                             dff_prenet=512,
                             max_position_encoding=10000,
                             postnet_conv_filters=32,
                             postnet_conv_layers=2,
                             postnet_kernel_size=5,
                             dropout_rate=0,
                             start_vec_value=-3,
                             end_vec_value=1,
                             debug=False):
    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=max_position_encoding,
        rate=dropout_rate,
    )
    
    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=max_position_encoding,
        rate=dropout_rate,
    )
    
    speech_out_module = SpeechPostnet(mel_channels=mel_channels,
                                      conv_filters=postnet_conv_filters,
                                      conv_layers=postnet_conv_layers,
                                      kernel_size=postnet_kernel_size)
    
    mel_text_transformer = TextMelTransformer(
        encoder_prenet=tf.keras.layers.Embedding(tokenizer.vocab_size, d_model),
        decoder_prenet=SpeechDecoderPrenet(d_model=d_model, dff=dff_prenet),
        decoder_postnet=speech_out_module,
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        start_vec_value=start_vec_value,
        end_vec_value=end_vec_value,
        debug=debug)
    
    return mel_text_transformer


def new_everything(tokenizer,
                   start_vec_value=-3,
                   end_vec_value=1,
                   mel_channels=80,
                   num_layers=1,
                   d_model=64,
                   num_heads=1,
                   dff=512,
                   dff_prenet=512,
                   max_position_encoding=10000,
                   postnet_conv_filters=32,
                   postnet_conv_layers=2,
                   postnet_kernel_size=5,
                   dropout_rate=0):
    speech_encoder_prenet = PointWiseFFN(d_model=d_model, dff=dff_prenet)
    speech_decoder_prenet = SpeechDecoderPrenet(d_model=d_model, dff=dff_prenet)
    speech_decoder_postnet = SpeechPostnet(mel_channels=mel_channels,
                                           conv_filters=postnet_conv_filters,
                                           conv_layers=postnet_conv_layers,
                                           kernel_size=postnet_kernel_size)
    speech_encoder = Encoder(num_layers=num_layers,
                             d_model=d_model,
                             num_heads=num_heads,
                             dff=dff,
                             maximum_position_encoding=max_position_encoding,
                             rate=dropout_rate)
    speech_decoder = Decoder(num_layers=num_layers,
                             d_model=d_model,
                             num_heads=num_heads,
                             dff=dff,
                             maximum_position_encoding=max_position_encoding,
                             rate=dropout_rate)
    
    text_encoder_prenet = tf.keras.layers.Embedding(tokenizer.vocab_size, d_model)
    text_decoder_prenet = tf.keras.layers.Embedding(tokenizer.vocab_size, d_model)
    text_decoder_postnet = TextPostnet(tokenizer.vocab_size)
    text_encoder = Encoder(num_layers=num_layers,
                           d_model=d_model,
                           num_heads=num_heads,
                           dff=dff,
                           maximum_position_encoding=max_position_encoding,
                           rate=dropout_rate, )
    text_decoder = Decoder(num_layers=num_layers,
                           d_model=d_model,
                           num_heads=num_heads,
                           dff=dff,
                           maximum_position_encoding=max_position_encoding,
                           rate=dropout_rate)
    
    transformers = {'text_to_mel': TextMelTransformer(encoder_prenet=text_encoder_prenet,
                                                      decoder_prenet=speech_decoder_prenet,
                                                      decoder_postnet=speech_decoder_postnet,
                                                      encoder=text_encoder,
                                                      decoder=speech_decoder,
                                                      tokenizer=tokenizer,
                                                      start_vec_value=start_vec_value,
                                                      end_vec_value=end_vec_value
                                                      ),
                    'mel_to_text': MelTextTransformer(encoder_prenet=speech_encoder_prenet,
                                                      decoder_prenet=text_decoder_prenet,
                                                      decoder_postnet=text_decoder_postnet,
                                                      encoder=speech_encoder,
                                                      decoder=text_decoder,
                                                      tokenizer=tokenizer,
                                                      mel_channels=mel_channels,
                                                      start_vec_value=start_vec_value,
                                                      end_vec_value=end_vec_value
                                                      ),
                    'mel_to_mel': MelTransformer(encoder_prenet=speech_encoder_prenet,
                                                 decoder_prenet=speech_decoder_prenet,
                                                 encoder=speech_encoder,
                                                 decoder=speech_decoder,
                                                 decoder_postnet=speech_decoder_postnet,
                                                 start_vec_value=start_vec_value,
                                                 end_vec_value=end_vec_value),
                    'text_to_text': TextTransformer(encoder_prenet=text_encoder_prenet,
                                                    decoder_prenet=text_decoder_prenet,
                                                    decoder_postnet=text_decoder_postnet,
                                                    encoder=text_encoder,
                                                    decoder=text_decoder,
                                                    tokenizer=tokenizer)}
    
    return transformers


def get_components(input_vocab_size,
                   target_vocab_size,
                   mel_channels=80,
                   num_layers=1,
                   d_model=64,
                   num_heads=1,
                   dff=512,
                   dff_prenet=512,
                   max_position_encoding=10000,
                   postnet_conv_filters=32,
                   postnet_conv_layers=2,
                   postnet_kernel_size=5,
                   dropout_rate=0):
    speech_encoder_prenet = PointWiseFFN(d_model=d_model, dff=dff_prenet)
    speech_decoder_prenet = SpeechDecoderPrenet(d_model=d_model, dff=dff_prenet)
    speech_decoder_postnet = SpeechPostnet(mel_channels=mel_channels,
                                           conv_filters=postnet_conv_filters,
                                           conv_layers=postnet_conv_layers,
                                           kernel_size=postnet_kernel_size)
    speech_encoder = Encoder(num_layers=num_layers,
                             d_model=d_model,
                             num_heads=num_heads,
                             dff=dff,
                             maximum_position_encoding=max_position_encoding,
                             rate=dropout_rate)
    speech_decoder = Decoder(num_layers=num_layers,
                             d_model=d_model,
                             num_heads=num_heads,
                             dff=dff,
                             maximum_position_encoding=max_position_encoding,
                             rate=dropout_rate)
    
    text_encoder_prenet = tf.keras.layers.Embedding(input_vocab_size, d_model)
    text_decoder_prenet = tf.keras.layers.Embedding(target_vocab_size, d_model)
    text_decoder_postnet = TextPostnet(target_vocab_size)
    text_encoder = Encoder(num_layers=num_layers,
                           d_model=d_model,
                           num_heads=num_heads,
                           dff=dff,
                           maximum_position_encoding=max_position_encoding,
                           rate=dropout_rate, )
    text_decoder = Decoder(num_layers=num_layers,
                           d_model=d_model,
                           num_heads=num_heads,
                           dff=dff,
                           maximum_position_encoding=max_position_encoding,
                           rate=dropout_rate)
    components = {'speech_encoder_prenet': speech_encoder_prenet,
                  'speech_decoder_prenet': speech_decoder_prenet,
                  'speech_decoder_postnet': speech_decoder_postnet,
                  'speech_encoder': speech_encoder,
                  'speech_decoder': speech_decoder,
                  'text_encoder_prenet': text_encoder_prenet,
                  'text_decoder_prenet': text_decoder_prenet,
                  'text_decoder_postnet': text_decoder_postnet,
                  'text_encoder': text_encoder,
                  'text_decoder': text_decoder}
    return components
