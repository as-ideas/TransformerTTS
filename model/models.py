import sys

import tensorflow as tf

from model.transformer_utils import create_encoder_padding_mask, create_mel_padding_mask, create_look_ahead_mask
from utils.losses import weighted_sum_losses
from model.layers import DecoderPrenet, Postnet, Decoder, Encoder
from utils.losses import masked_mean_absolute_error, new_scaled_crossentropy
from preprocessing.data_handling import Tokenizer
from preprocessing.text_processing import _phonemes, Phonemizer, _punctuations
from model.layers import DurationPredictor, Expand, ConvEncoder, ConvDecoder


class AutoregressiveTransformer(tf.keras.models.Model):
    
    def __init__(self,
                 mel_channels: int,
                 encoder_model_dimension: int,
                 encoder_num_heads: list,
                 encoder_feed_forward_dimension: int,
                 decoder_model_dimension: int,
                 decoder_prenet_dimension: int,
                 decoder_num_heads: list,
                 decoder_feed_forward_dimension: int,
                 postnet_conv_filters: int,
                 postnet_conv_layers: int,
                 postnet_kernel_size: int,
                 max_position_encoding: int,
                 dropout_rate: float,
                 mel_start_value: int,
                 mel_end_value: int,
                 max_r: int = 10,
                 phoneme_language: str = 'en',
                 decoder_prenet_dropout=0.,
                 debug=False,
                 **kwargs):
        super(AutoregressiveTransformer, self).__init__(**kwargs)
        self.start_vec = tf.ones((1, mel_channels), dtype=tf.float32) * mel_start_value
        self.end_vec = tf.ones((1, mel_channels), dtype=tf.float32) * mel_end_value
        self.stop_prob_index = 2
        self.max_r = max_r
        self.r = max_r
        self.mel_channels = mel_channels
        self.decoder_prenet_dropout = decoder_prenet_dropout
        
        self.tokenizer = Tokenizer(sorted(list(_phonemes) + list(_punctuations)), add_start_end=True)
        self.phonemizer = Phonemizer(language=phoneme_language)
        
        self.encoder_prenet = tf.keras.layers.Embedding(self.tokenizer.vocab_size, encoder_model_dimension,
                                                        name='Embedding')
        self.encoder = Encoder(model_dim=encoder_model_dimension,
                               num_heads=encoder_num_heads,
                               dense_hidden_units=encoder_feed_forward_dimension,
                               maximum_position_encoding=max_position_encoding,
                               dropout_rate=dropout_rate,
                               name='Encoder')
        self.decoder_prenet = DecoderPrenet(model_dim=decoder_model_dimension,
                                            dense_hidden_units=decoder_prenet_dimension,
                                            name='DecoderPrenet')
        self.decoder = Decoder(model_dim=decoder_model_dimension,
                               num_heads=decoder_num_heads,
                               dense_hidden_units=decoder_feed_forward_dimension,
                               maximum_position_encoding=max_position_encoding,
                               dropout_rate=dropout_rate,
                               name='Decoder')
        self.final_proj_mel = tf.keras.layers.Dense(self.mel_channels * self.max_r, name='FinalProj')
        self.decoder_postnet = Postnet(mel_channels=mel_channels,
                                       conv_filters=postnet_conv_filters,
                                       conv_layers=postnet_conv_layers,
                                       kernel_size=postnet_kernel_size,
                                       name='Postnet')
        
        self.training_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32)
        ]
        self.forward_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32)
        ]
        self.encoder_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32)
        ]
        self.decoder_signature = [
            tf.TensorSpec(shape=(None, None, encoder_model_dimension), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
        ]
        self.debug = debug
        self.__apply_all_signatures()
    
    def __apply_all_signatures(self):
        self.forward = self.__apply_signature(self._forward, self.forward_input_signature)
        self.train_step = self.__apply_signature(self._train_step, self.training_input_signature)
        self.val_step = self.__apply_signature(self._val_step, self.training_input_signature)
        self.forward_encoder = self.__apply_signature(self._forward_encoder, self.encoder_signature)
        self.forward_decoder = self.__apply_signature(self._forward_decoder, self.decoder_signature)
    
    def __apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)
    
    def _call_encoder(self, inputs, training):
        padding_mask = create_encoder_padding_mask(inputs)
        enc_input = self.encoder_prenet(inputs)
        enc_output = self.encoder(inputs=enc_input,
                                  training=training,
                                  mask=padding_mask)
        return enc_output, padding_mask
    
    def _call_decoder(self, encoder_output, targets, encoder_padding_mask, training):
        dec_target_padding_mask = create_mel_padding_mask(targets)
        look_ahead_mask = create_look_ahead_mask(tf.shape(targets)[1])
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        dec_input = self.decoder_prenet(targets, training=training, dropout_rate=self.decoder_prenet_dropout)
        dec_output, attention_weights = self.decoder(inputs=dec_input,
                                                     enc_output=encoder_output,
                                                     training=training,
                                                     look_ahead_mask=combined_mask,
                                                     padding_mask=encoder_padding_mask)
        out_proj = self.final_proj_mel(dec_output)[:, :, :self.r * self.mel_channels]
        b = int(tf.shape(out_proj)[0])
        t = int(tf.shape(out_proj)[1])
        mel = tf.reshape(out_proj, (b, t * self.r, self.mel_channels))
        model_output = self.decoder_postnet(inputs=mel, training=training)
        model_output.update(
            {'attention_weights': attention_weights, 'decoder_output': dec_output, 'out_proj': out_proj})
        return model_output
    
    def call(self, inputs, targets, training):
        encoder_output, padding_mask = self._call_encoder(inputs, training)
        model_out = self._call_decoder(encoder_output, targets, padding_mask, training)
        return model_out
    
    def predict(self, inp, max_length=1000, encode=True, verbose=True):
        if encode:
            inp = self.encode_text(inp)
        inp = tf.cast(tf.expand_dims(inp, 0), tf.int32)
        output = tf.cast(tf.expand_dims(self.start_vec, 0), tf.float32)
        output_concat = tf.cast(tf.expand_dims(self.start_vec, 0), tf.float32)
        out_dict = {}
        encoder_output, padding_mask = self.forward_encoder(inp)
        for i in range(int(max_length // self.r) + 1):
            model_out = self.forward_decoder(encoder_output, output, padding_mask)
            output = tf.concat([output, model_out['final_output'][:1, -1:, :]], axis=-2)
            output_concat = tf.concat([tf.cast(output_concat, tf.float32), model_out['final_output'][:1, -self.r:, :]],
                                      axis=-2)
            stop_pred = model_out['stop_prob'][:, -1]
            out_dict = {'mel': output_concat[0, 1:, :], 'attention_weights': model_out['attention_weights']}
            if verbose:
                sys.stdout.write(f'\rpred text mel: {i} stop out: {float(stop_pred[0, 2])}')
            if int(tf.argmax(stop_pred, axis=-1)) == self.stop_prob_index:
                if verbose:
                    print('Stopping')
                break
        return out_dict
    
    def _set_r(self, r):
        if self.r == r:
            return
        self.r = r
        self.__apply_all_signatures()
    
    def set_constants(self, decoder_prenet_dropout: float = None, learning_rate: float = None,
                      reduction_factor: float = None):
        if decoder_prenet_dropout is not None:
            self.decoder_prenet_dropout = decoder_prenet_dropout
        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)
        if reduction_factor is not None:
            self._set_r(reduction_factor)
    
    def _forward(self, inp, output):
        model_out = self.__call__(inputs=inp,
                                  targets=output,
                                  training=False)
        return model_out
    
    def _forward_encoder(self, inputs):
        return self._call_encoder(inputs, training=False)
    
    def _forward_decoder(self, encoder_output, targets, encoder_padding_mask):
        return self._call_decoder(encoder_output, targets, encoder_padding_mask, training=False)
    
    def _gta_forward(self, inp, tar, stop_prob, training):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        tar_stop_prob = stop_prob[:, 1:]
        
        mel_len = int(tf.shape(tar_inp)[1])
        tar_mel = tar_inp[:, 0::self.r, :]
        
        with tf.GradientTape() as tape:
            model_out = self.__call__(inputs=inp,
                                      targets=tar_mel,
                                      training=training)
            loss, loss_vals = weighted_sum_losses((tar_real,
                                                   tar_stop_prob,
                                                   tar_real),
                                                  (model_out['final_output'][:, :mel_len, :],
                                                   model_out['stop_prob'][:, :mel_len, :],
                                                   model_out['mel_linear'][:, :mel_len, :]),
                                                  self.loss,
                                                  self.loss_weights)
        model_out.update({'loss': loss})
        model_out.update({'losses': {'output': loss_vals[0], 'stop_prob': loss_vals[1], 'mel_linear': loss_vals[2]}})
        model_out.update({'reduced_target': tar_mel})
        return model_out, tape
    
    def _train_step(self, inp, tar, stop_prob):
        model_out, tape = self._gta_forward(inp, tar, stop_prob, training=True)
        gradients = tape.gradient(model_out['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out
    
    def _val_step(self, inp, tar, stop_prob):
        model_out, _ = self._gta_forward(inp, tar, stop_prob, training=False)
        return model_out
    
    @property
    def step(self):
        return int(self.optimizer.iterations)
    
    def _compile(self, stop_scaling, optimizer):
        self.compile(loss=[masked_mean_absolute_error,
                           new_scaled_crossentropy(index=2, scaling=stop_scaling),
                           masked_mean_absolute_error],
                     loss_weights=[1., 1., 1.],
                     optimizer=optimizer)
    
    def build_graph(self, r: int):
        self._set_r(r)
        try:
            self.forward([0], output=[0], decoder_prenet_dropout=0)
        except:
            pass
    
    def load_weights(self, weights_path: str, r: int = 1):
        self.build_graph(r)
        super(AutoregressiveTransformer, self).load_weights(weights_path)
    
    def load_checkpoint(self, checkpoint_dir: str, checkpoint_path: str = None, r: int = 1):
        self.build_graph(self.max_r)
        ckpt = tf.train.Checkpoint(net=self)
        manager = tf.train.CheckpointManager(ckpt, checkpoint_dir,
                                             max_to_keep=None)
        if checkpoint_path:
            ckpt.restore(checkpoint_path)
        else:
            ckpt.restore(manager.latest_checkpoint)
        self._set_r(r)
        return ckpt, manager
    
    def encode_text(self, text):
        phons = self.phonemizer.encode(text, clean=True)
        return self.tokenizer.encode(phons)


class ForwardTransformer(tf.keras.models.Model):
    def __init__(self, model_dim: int, dropout_rate: float, decoder_num_heads: list,
                 encoder_num_heads: list, encoder_maximum_postion_encoding: int, decoder_maximum_position_encoding: int,
                 encoder_feed_forward_dimension: int = None,
                 encoder_dense_blocks=1, mel_channels=80, phoneme_language='en', debug=False, **kwargs):
        super(ForwardTransformer, self).__init__(**kwargs)
        self.tokenizer = Tokenizer(sorted(list(_phonemes) + list(_punctuations)), add_start_end=False)
        self.phonemizer = Phonemizer(language=phoneme_language)
        self.encoder_prenet = tf.keras.layers.Embedding(self.tokenizer.vocab_size, model_dim,
                                                        name='Embedding')
        self.encoder = ConvEncoder(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=encoder_num_heads,
                                   feed_forward_dimension=encoder_feed_forward_dimension,
                                   maximum_position_encoding=encoder_maximum_postion_encoding,
                                   dense_blocks=encoder_dense_blocks)
        self.dur_pred = DurationPredictor(model_dim=model_dim, dropout_rate=dropout_rate, name='dur_pred')
        self.expand = Expand(name='expand', model_dim=model_dim)
        self.decoder = ConvDecoder(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=decoder_num_heads,
                                   maximum_position_encoding=decoder_maximum_position_encoding,
                                   mel_channels=mel_channels)
        self.training_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32)
        ]
        self.forward_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        ]
        self.debug = debug
        self.__apply_all_signatures()
    
    def __apply_all_signatures(self):
        self.forward = self.__apply_signature(self._forward, self.forward_input_signature)
        self.train_step = self.__apply_signature(self._train_step, self.training_input_signature)
        self.val_step = self.__apply_signature(self._val_step, self.training_input_signature)
    
    def call(self, x, target_durations, training):
        padding_mask = create_encoder_padding_mask(x)
        x = self.encoder_prenet(x)
        x = self.encoder(x, training=training, padding_mask=padding_mask)
        durations = self.dur_pred(x, training=training)
        if target_durations is not None:
            mels = self.expand(x, target_durations)
        else:
            mels = self.expand(x, durations)
        expanded_mask = create_mel_padding_mask(mels)
        mels = self.decoder(mels, training=training, padding_mask=expanded_mask)
        model_out = {'mel': mels,
                     'duration': durations,
                     'expanded mask': expanded_mask}
        return model_out
    
    def _train_step(self, input_sequence, target_sequence, target_durations):
        target_durations = tf.expand_dims(target_durations, -1)
        with tf.GradientTape() as tape:
            model_out = self.__call__(input_sequence, target_durations, training=True)
            loss, loss_vals = weighted_sum_losses((target_sequence,
                                                   target_durations),
                                                  (model_out['mel'],
                                                   model_out['duration']),
                                                  self.loss,
                                                  self.loss_weights)
        model_out.update({'loss': loss})
        model_out.update({'losses': {'mel': loss_vals[0], 'duration': loss_vals[1]}})
        gradients = tape.gradient(model_out['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out
    
    def build_graph(self):
        try:
            self.forward([0], output=[0], decoder_prenet_dropout=0)
        except:
            pass
    
    def load_weights(self, weights_path: str):
        self.build_graph()
        self.load_weights(weights_path)
    
    def load_checkpoint(self, checkpoint_dir: str, checkpoint_path: str = None):
        self.build_graph()
        ckpt = tf.train.Checkpoint(net=self)
        manager = tf.train.CheckpointManager(ckpt, checkpoint_dir,
                                             max_to_keep=None)
        if checkpoint_path:
            ckpt.restore(checkpoint_path)
        else:
            ckpt.restore(manager.latest_checkpoint)
        return ckpt, manager
    
    @property
    def step(self):
        return int(self.optimizer.iterations)
    
    def set_constants(self, learning_rate: float = None):
        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)
    
    def __apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)
    
    def encode_text(self, text):
        phons = self.phonemizer.encode(text, clean=True)
        return self.tokenizer.encode(phons)
    
    def _compile(self, optimizer):
        self.compile(loss=[masked_mean_absolute_error,
                           masked_mean_absolute_error],
                     loss_weights=[1., 1.],
                     optimizer=optimizer)
    
    def _val_step(self, input_sequence, target_sequence, target_durations):
        target_durations = tf.expand_dims(target_durations, -1)
        model_out = self.__call__(input_sequence, target_durations, training=False)
        loss, loss_vals = weighted_sum_losses((target_sequence,
                                               target_durations),
                                              (model_out['mel'],
                                               model_out['duration']),
                                              self.loss,
                                              self.loss_weights)
        model_out.update({'loss': loss})
        model_out.update({'losses': {'mel': loss_vals[0], 'duration': loss_vals[1]}})
        return model_out
    
    def _forward(self, input_sequence):
        return self.__call__(input_sequence, target_durations=None, training=False)
    
    def predict(self, inp, encode=True):
        if encode:
            inp = self.encode_text(inp)
            inp = tf.cast(tf.expand_dims(inp, 0), tf.int32)
        return self.forward(inp)
