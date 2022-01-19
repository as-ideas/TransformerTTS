from pathlib import Path
import subprocess

import tensorflow as tf
import numpy as np
from ruamel.yaml import YAML

from model.transformer_utils import create_encoder_padding_mask, create_mel_padding_mask, create_look_ahead_mask
from utils.losses import weighted_sum_losses, masked_mean_absolute_error, new_scaled_crossentropy
from data.text import TextToTokens
from model.layers import DecoderPrenet, Postnet, StatPredictor, Expand, SelfAttentionBlocks, CrossAttentionBlocks
from utils.metrics import batch_diagonal_mask


class Aligner(tf.keras.models.Model):
    
    def __init__(self,
                 encoder_model_dimension: int,
                 decoder_model_dimension: int,
                 encoder_num_heads: list,
                 decoder_num_heads: list,
                 encoder_max_position_encoding: int,
                 decoder_max_position_encoding: int,
                 encoder_prenet_dimension: int,
                 decoder_prenet_dimension: int,
                 dropout_rate: float,
                 mel_start_value: float,
                 mel_end_value: float,
                 mel_channels: int,
                 phoneme_language: str,
                 with_stress: bool,
                 decoder_prenet_dropout: int,
                 model_breathing: bool,
                 encoder_feed_forward_dimension: int = None,
                 decoder_feed_forward_dimension: int = None,
                 max_r: int = 10,
                 debug=False,
                 **kwargs):
        super(Aligner, self).__init__(**kwargs)
        self.config = self._make_config(locals())
        self.start_vec = tf.ones((1, mel_channels), dtype=tf.float32) * mel_start_value
        self.end_vec = tf.ones((1, mel_channels), dtype=tf.float32) * mel_end_value
        self.stop_prob_index = 2
        self.max_r = max_r
        self.r = max_r
        self.mel_channels = mel_channels
        self.force_encoder_diagonal = False
        self.force_decoder_diagonal = False
        self.text_pipeline = TextToTokens.default(phoneme_language,
                                                  add_start_end=True,
                                                  with_stress=with_stress,
                                                  model_breathing=model_breathing)
        self.encoder_prenet = tf.keras.layers.Embedding(self.text_pipeline.tokenizer.vocab_size,
                                                        encoder_prenet_dimension,
                                                        name='Embedding')
        self.encoder = SelfAttentionBlocks(model_dim=encoder_model_dimension,
                                           dropout_rate=dropout_rate,
                                           num_heads=encoder_num_heads,
                                           feed_forward_dimension=encoder_feed_forward_dimension,
                                           maximum_position_encoding=encoder_max_position_encoding,
                                           dense_blocks=len(encoder_num_heads),
                                           conv_filters=None,
                                           kernel_size=None,
                                           conv_activation=None,
                                           name='Encoder')
        self.decoder_prenet = DecoderPrenet(model_dim=decoder_model_dimension,
                                            dense_hidden_units=decoder_prenet_dimension,
                                            dropout_rate=decoder_prenet_dropout,
                                            name='DecoderPrenet')
        self.decoder = CrossAttentionBlocks(model_dim=decoder_model_dimension,
                                            dropout_rate=dropout_rate,
                                            num_heads=decoder_num_heads,
                                            feed_forward_dimension=decoder_feed_forward_dimension,
                                            maximum_position_encoding=decoder_max_position_encoding,
                                            name='Decoder')
        self.final_proj_mel = tf.keras.layers.Dense(self.mel_channels * self.max_r, name='FinalProj')
        self.decoder_postnet = Postnet(mel_channels=mel_channels,
                                       name='Postnet')
        
        self.training_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32)
        ]
        self.forward_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
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
        self._apply_all_signatures()
    
    @property
    def step(self):
        return int(self.optimizer.iterations)
    
    def _apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)
    
    def _apply_all_signatures(self):
        self.forward = self._apply_signature(self._forward, self.forward_input_signature)
        self.train_step = self._apply_signature(self._train_step, self.training_input_signature)
        self.val_step = self._apply_signature(self._val_step, self.training_input_signature)
        self.forward_encoder = self._apply_signature(self._forward_encoder, self.encoder_signature)
        self.forward_decoder = self._apply_signature(self._forward_decoder, self.decoder_signature)
        
    def _make_config(self, locals) -> dict:
        config = {}
        for k in locals:
            if (k != 'self') and (k != '__class__'):
                if isinstance(locals[k], dict):
                    config.update(locals[k])
                else:
                    config.update({k: locals[k]})
        return dict(config)
    
    def _call_encoder(self, inputs, training):
        padding_mask = create_encoder_padding_mask(inputs)
        enc_input = self.encoder_prenet(inputs)
        enc_output, attn_weights = self.encoder(enc_input,
                                                training=training,
                                                padding_mask=padding_mask)
        return enc_output, padding_mask, attn_weights
    
    def _call_decoder(self, encoder_output, targets, encoder_padding_mask, training):
        dec_target_padding_mask = create_mel_padding_mask(targets)
        look_ahead_mask = create_look_ahead_mask(tf.shape(targets)[1])
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        dec_input = self.decoder_prenet(targets, training=training)
        dec_output, attention_weights = self.decoder(inputs=dec_input,
                                                     enc_output=encoder_output,
                                                     training=training,
                                                     decoder_padding_mask=combined_mask,
                                                     encoder_padding_mask=encoder_padding_mask,
                                                     reduction_factor=self.r)
        out_proj = self.final_proj_mel(dec_output)[:, :, :self.r * self.mel_channels]
        b = int(tf.shape(out_proj)[0])
        t = int(tf.shape(out_proj)[1])
        mel = tf.reshape(out_proj, (b, t * self.r, self.mel_channels))
        model_output = self.decoder_postnet(mel)
        model_output.update(
            {'decoder_attention': attention_weights, 'decoder_output': dec_output, 'linear': mel,
             'mel_mask': dec_target_padding_mask})
        return model_output
    
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
                                                   tar_stop_prob),
                                                  (model_out['mel'][:, :mel_len, :],
                                                   model_out['stop_prob'][:, :mel_len, :]),
                                                  self.loss,
                                                  self.loss_weights)
            
            phon_len = tf.reduce_sum(1. - tf.squeeze(model_out['text_mask'], axis=(1, 2)), axis=1)
            d_loss = 0.
            norm_factor = 1.
            if self.force_decoder_diagonal:
                mel_len = tf.reduce_sum(1. - tf.squeeze(model_out['mel_mask'], axis=(1, 2)), axis=1)
                dec_key_list = list(model_out['decoder_attention'].keys())
                decoder_dmask = batch_diagonal_mask(model_out['decoder_attention'][dec_key_list[0]], mel_len, phon_len)
                for key in dec_key_list:
                    d_measure = tf.reduce_sum(model_out['decoder_attention'][key] * decoder_dmask, axis=(-2, -1))
                    d_loss += tf.reduce_mean(d_measure) / 10.
                norm_factor += len(model_out['decoder_attention'].keys())
            
            if self.force_encoder_diagonal:
                enc_key_list = list(model_out['encoder_attention'].keys())
                encoder_dmask = batch_diagonal_mask(model_out['encoder_attention'][enc_key_list[0]], phon_len, phon_len)
                for key in enc_key_list:
                    d_measure = tf.reduce_sum(model_out['encoder_attention'][key] * encoder_dmask, axis=(-2, -1))
                    d_loss += tf.reduce_mean(d_measure) / 10.
                norm_factor += len(model_out['encoder_attention'].keys())
            d_loss /= norm_factor
            loss += d_loss
        model_out.update({'loss': loss})
        model_out.update({'losses': {'mel': loss_vals[0], 'stop_prob': loss_vals[1], 'diag_loss': d_loss}})
        return model_out, tape
    
    def _train_step(self, inp, tar, stop_prob):
        model_out, tape = self._gta_forward(inp, tar, stop_prob, training=True)
        gradients = tape.gradient(model_out['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out
    
    def _val_step(self, inp, tar, stop_prob):
        model_out, _ = self._gta_forward(inp, tar, stop_prob, training=False)
        return model_out
    
    def _compile(self, stop_scaling, optimizer):
        self.loss_weights = [1., 1.]
        self.compile(loss=[masked_mean_absolute_error,
                           new_scaled_crossentropy(index=2, scaling=stop_scaling)],
                     loss_weights=self.loss_weights,
                     optimizer=optimizer)
    
    def _set_r(self, r):
        if self.r == r:
            return
        self.r = r
        self._apply_all_signatures()
    
    def _set_force_encoder_diagonal(self, value):
        if self.force_encoder_diagonal == value:
            return
        self.force_encoder_diagonal = value
        self._apply_all_signatures()
    
    def _set_force_decoder_diagonal(self, value):
        if self.force_decoder_diagonal == value:
            return
        self.force_decoder_diagonal = value
        self._apply_all_signatures()
    
    def align(self, text, mel, mels_have_start_end_vectors=False, phonemize=False, encode_phonemes=False, plot=True):
        if phonemize:
            text = self.text_pipeline.phonemizer(text)
        if encode_phonemes:
            text = self.text_pipeline.tokenizer(text)
        
        if len(tf.shape(text)) < 2:
            text = tf.expand_dims(text, axis=0)
        text = tf.cast(text, tf.int32)
        if len(tf.shape(mel)) < 3:
            mel = tf.expand_dims(mel, axis=0)
        if self.r != 1:
            print('WARNING: reduction factor != 1.')
        if mels_have_start_end_vectors:
            tar_inp = mel[:, :-1]
        else:
            start_vecs = tf.expand_dims(self.start_vec, axis=0)
            start_vecs = tf.tile(start_vecs, (tf.shape(mel)[0], 1, 1))
            tar_inp = np.concatenate([start_vecs, mel], axis=1)
        autoregr_tar_mel = tar_inp[:, 0::self.r, :]
        model_out = self.forward(inp=text, output=autoregr_tar_mel)
        attn_weights = model_out['decoder_attention']['Decoder_LastBlock_CrossAttention']
        return attn_weights, model_out
    
    def predict(self, inp, max_length=1000, encode=True, verbose=True):
        if encode:
            inp = self.encode_text(inp)
        inp = tf.cast(tf.expand_dims(inp, 0), tf.int32)
        output = tf.cast(tf.expand_dims(self.start_vec, 0), tf.float32)
        output_concat = tf.cast(tf.expand_dims(self.start_vec, 0), tf.float32)
        out_dict = {}
        encoder_output, padding_mask, encoder_attention = self.forward_encoder(inp)
        for i in range(int(max_length // self.r) + 1):
            model_out = self.forward_decoder(encoder_output, output, padding_mask)
            output = tf.concat([output, model_out['mel'][:1, -1:, :]], axis=-2)
            output_concat = tf.concat([tf.cast(output_concat, tf.float32), model_out['mel'][:1, -self.r:, :]],
                                      axis=-2)
            stop_pred = model_out['stop_prob'][:, -1]
            out_dict = {'mel': output_concat[0, 1:, :],
                        'decoder_attention': model_out['decoder_attention'],
                        'encoder_attention': encoder_attention}
            if int(tf.argmax(stop_pred, axis=-1)) == self.stop_prob_index:
                if verbose:
                    print('Stopping')
                break
        return out_dict
    
    def call(self, inputs, targets, training):
        encoder_output, padding_mask, encoder_attention = self._call_encoder(inputs, training)
        model_out = self._call_decoder(encoder_output, targets, padding_mask, training)
        model_out.update({'encoder_attention': encoder_attention, 'text_mask': padding_mask})
        return model_out
    
    def set_constants(self,
                      learning_rate: float = None,
                      reduction_factor: float = None,
                      force_encoder_diagonal: bool = None,
                      force_decoder_diagonal: bool = None):
        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)
        if reduction_factor is not None:
            self._set_r(reduction_factor)
        if force_encoder_diagonal is not None:
            self._set_force_encoder_diagonal(force_encoder_diagonal)
        if force_decoder_diagonal is not None:
            self._set_force_decoder_diagonal(force_decoder_diagonal)
    
    def encode_text(self, text):
        return self.text_pipeline(text)
    
    def build_model_weights(self) -> None:
        _ = self(tf.zeros((1, 1)), tf.zeros((1, 1, self.mel_channels)), training=False)
    
    @classmethod
    def from_config(cls, config, max_r=10):
        return cls(mel_channels=config['mel_channels'],
                   encoder_model_dimension=config['encoder_model_dimension'],
                   decoder_model_dimension=config['decoder_model_dimension'],
                   encoder_num_heads=config['encoder_num_heads'],
                   decoder_num_heads=config['decoder_num_heads'],
                   encoder_feed_forward_dimension=config['encoder_feed_forward_dimension'],
                   decoder_feed_forward_dimension=config['decoder_feed_forward_dimension'],
                   encoder_max_position_encoding=config['encoder_max_position_encoding'],
                   decoder_max_position_encoding=config['decoder_max_position_encoding'],
                   decoder_prenet_dimension=config['decoder_prenet_dimension'],
                   encoder_prenet_dimension=config['encoder_prenet_dimension'],
                   dropout_rate=config['dropout_rate'],
                   decoder_prenet_dropout=config['decoder_prenet_dropout'],
                   max_r=max_r,
                   mel_start_value=config['mel_start_value'],
                   mel_end_value=config['mel_end_value'],
                   phoneme_language=config['phoneme_language'],
                   with_stress=config['with_stress'],
                   debug=config['debug'],
                   model_breathing=config['model_breathing'])


class ForwardTransformer(tf.keras.models.Model):
    def __init__(self,
                 encoder_model_dimension: int,
                 decoder_model_dimension: int,
                 dropout_rate: float,
                 decoder_num_heads: list,
                 encoder_num_heads: list,
                 encoder_max_position_encoding: int,
                 decoder_max_position_encoding: int,
                 encoder_dense_blocks: int,
                 decoder_dense_blocks: int,
                 duration_conv_filters: list,
                 pitch_conv_filters: list,
                 duration_kernel_size: int,
                 pitch_kernel_size: int,
                 predictors_dropout: float,
                 mel_channels: int,
                 phoneme_language: str,
                 with_stress: bool,
                 model_breathing: bool,
                 transposed_attn_convs: bool,
                 encoder_attention_conv_filters: list = None,
                 decoder_attention_conv_filters: list = None,
                 encoder_attention_conv_kernel: int = None,
                 decoder_attention_conv_kernel: int = None,
                 encoder_feed_forward_dimension: int = None,
                 decoder_feed_forward_dimension: int = None,
                 debug=False,
                 **kwargs):
        super(ForwardTransformer, self).__init__()
        self.config = self._make_config(locals(), kwargs)
        self.text_pipeline = TextToTokens.default(phoneme_language,
                                                  add_start_end=False,
                                                  with_stress=with_stress,
                                                  model_breathing=model_breathing)
        self.symbols = self.text_pipeline.tokenizer.alphabet
        self.mel_channels = mel_channels
        self.encoder_prenet = tf.keras.layers.Embedding(self.text_pipeline.tokenizer.vocab_size,
                                                        encoder_model_dimension,
                                                        name='Embedding')
        self.encoder = SelfAttentionBlocks(model_dim=encoder_model_dimension,
                                           dropout_rate=dropout_rate,
                                           num_heads=encoder_num_heads,
                                           feed_forward_dimension=encoder_feed_forward_dimension,
                                           maximum_position_encoding=encoder_max_position_encoding,
                                           dense_blocks=encoder_dense_blocks,
                                           conv_filters=encoder_attention_conv_filters,
                                           kernel_size=encoder_attention_conv_kernel,
                                           conv_activation='relu',
                                           transposed_convs=transposed_attn_convs,
                                           name='Encoder')
        self.dur_pred = StatPredictor(conv_filters=duration_conv_filters,
                                      kernel_size=duration_kernel_size,
                                      conv_padding='same',
                                      conv_activation='relu',
                                      dense_activation='relu',
                                      dropout_rate=predictors_dropout,
                                      name='dur_pred')
        self.expand = Expand(name='expand', model_dim=encoder_model_dimension)
        self.pitch_pred = StatPredictor(conv_filters=pitch_conv_filters,
                                        kernel_size=pitch_kernel_size,
                                        conv_padding='same',
                                        conv_activation='relu',
                                        dense_activation='linear',
                                        dropout_rate=predictors_dropout,
                                        name='pitch_pred')
        self.pitch_embed = tf.keras.layers.Dense(encoder_model_dimension, activation='relu')
        self.decoder = SelfAttentionBlocks(model_dim=decoder_model_dimension,
                                           dropout_rate=dropout_rate,
                                           num_heads=decoder_num_heads,
                                           feed_forward_dimension=decoder_feed_forward_dimension,
                                           maximum_position_encoding=decoder_max_position_encoding,
                                           dense_blocks=decoder_dense_blocks,
                                           conv_filters=decoder_attention_conv_filters,
                                           kernel_size=decoder_attention_conv_kernel,
                                           conv_activation='relu',
                                           transposed_convs=transposed_attn_convs,
                                           name='Decoder')
        self.out = tf.keras.layers.Dense(mel_channels)
        self.training_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32)
        ]
        self.forward_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ]
        self.forward_masked_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        ]
        self.debug = debug
        self._apply_all_signatures()
    
    def _apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)
    
    def _apply_all_signatures(self):
        self.forward = self._apply_signature(self._forward, self.forward_input_signature)
        self.train_step = self._apply_signature(self._train_step, self.training_input_signature)
        self.val_step = self._apply_signature(self._val_step, self.training_input_signature)
    
    def _make_config(self, locals:dict, kwargs:dict) -> dict:
        config = {}
        keys = [k for k in locals.keys() if (k not in kwargs) and (k not in ['self', '__class__', 'kwargs'] )]
        for k in keys:
            if isinstance(locals[k], dict):
                config.update(locals[k])
            else:
                config.update({k: locals[k]})
        config.update(kwargs)
        return config
    
    def _train_step(self, input_sequence, target_sequence, target_durations, target_pitch):
        target_durations = tf.expand_dims(target_durations, -1)
        target_pitch = tf.expand_dims(target_pitch, -1)
        mel_len = int(tf.shape(target_sequence)[1])
        with tf.GradientTape() as tape:
            model_out = self.__call__(input_sequence, target_durations, target_pitch=target_pitch, training=True)
            loss, loss_vals = weighted_sum_losses((target_sequence,
                                                   target_durations,
                                                   target_pitch),
                                                  (model_out['mel'][:, :mel_len, :],
                                                   model_out['duration'],
                                                   model_out['pitch']),
                                                  self.loss,
                                                  self.loss_weights)
        model_out.update({'loss': loss})
        model_out.update({'losses': {'mel': loss_vals[0], 'duration': loss_vals[1], 'pitch': loss_vals[2]}})
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out
    
    def _compile(self, optimizer):
        self.loss_weights = [1., 1., 3.]
        self.compile(loss=[masked_mean_absolute_error,
                           masked_mean_absolute_error,
                           masked_mean_absolute_error],
                     loss_weights=self.loss_weights,
                     optimizer=optimizer)
    
    def _val_step(self, input_sequence, target_sequence, target_durations, target_pitch):
        target_durations = tf.expand_dims(target_durations, -1)
        target_pitch = tf.expand_dims(target_pitch, -1)
        mel_len = int(tf.shape(target_sequence)[1])
        model_out = self.__call__(input_sequence, target_durations, target_pitch=target_pitch, training=False)
        loss, loss_vals = weighted_sum_losses((target_sequence,
                                               target_durations,
                                               target_pitch),
                                              (model_out['mel'][:, :mel_len, :],
                                               model_out['duration'],
                                               model_out['pitch']),
                                              self.loss,
                                              self.loss_weights)
        model_out.update({'loss': loss})
        model_out.update({'losses': {'mel': loss_vals[0], 'duration': loss_vals[1], 'pitch': loss_vals[2]}})
        return model_out
    
    def _forward(self, input_sequence, durations_scalar):
        return self.__call__(input_sequence, target_durations=None, target_pitch=None, training=False,
                             durations_scalar=durations_scalar, max_durations_mask=None,
                             min_durations_mask=None)
    
    @property
    def step(self):
        return int(self.optimizer.iterations)
    
    def call(self, x, target_durations=None, target_pitch=None, training=False, durations_scalar=1.,
             max_durations_mask=None,
             min_durations_mask=None):
        encoder_padding_mask = create_encoder_padding_mask(x)
        x = self.encoder_prenet(x)
        x, encoder_attention = self.encoder(x, training=training, padding_mask=encoder_padding_mask)
        padding_mask = 1. - tf.squeeze(encoder_padding_mask, axis=(1, 2))[:, :, None]
        durations = self.dur_pred(x, training=training, mask=padding_mask)
        pitch = self.pitch_pred(x, training=training, mask=padding_mask)
        if target_pitch is not None:
            pitch_embed = self.pitch_embed(target_pitch)
        else:
            pitch_embed = self.pitch_embed(pitch)
        x = x + pitch_embed
        if target_durations is not None:
            use_durations = target_durations
        else:
            use_durations = durations * durations_scalar
        if max_durations_mask is not None:
            use_durations = tf.math.minimum(use_durations, tf.expand_dims(max_durations_mask, -1))
        if min_durations_mask is not None:
            use_durations = tf.math.maximum(use_durations, tf.expand_dims(min_durations_mask, -1))
        mels = self.expand(x, use_durations)
        expanded_mask = create_mel_padding_mask(mels)
        mels, decoder_attention = self.decoder(mels, training=training, padding_mask=expanded_mask, reduction_factor=1)
        mels = self.out(mels)
        model_out = {'mel': mels,
                     'duration': durations,
                     'pitch': pitch,
                     'expanded_mask': expanded_mask,
                     'encoder_attention': encoder_attention,
                     'decoder_attention': decoder_attention}
        return model_out
    
    def set_constants(self, learning_rate: float = None, **kwargs):
        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)
    
    def encode_text(self, text):
        return self.text_pipeline(text)
    
    def predict(self, inp, encode=True, speed_regulator=1., phoneme_max_duration=None, phoneme_min_duration=None,
                max_durations_mask=None, min_durations_mask=None, phoneme_durations=None, phoneme_pitch=None):
        if encode:
            inp = self.encode_text(inp)
        if len(tf.shape(inp)) < 2:
            inp = tf.expand_dims(inp, 0)
        inp = tf.cast(inp, tf.int32)
        duration_scalar = tf.cast(1. / speed_regulator, tf.float32)
        max_durations_mask = self._make_max_duration_mask(inp, phoneme_max_duration)
        min_durations_mask = self._make_min_duration_mask(inp, phoneme_min_duration)
        out = self.call(inp,
                        target_durations=phoneme_durations,
                        target_pitch=phoneme_pitch,
                        training=False,
                        durations_scalar=duration_scalar,
                        max_durations_mask=max_durations_mask,
                        min_durations_mask=min_durations_mask)
        out['mel'] = tf.squeeze(out['mel'])
        return out
    
    def _make_max_duration_mask(self, encoded_text, phoneme_max_duration):
        np_text = np.array(encoded_text)
        new_mask = np.ones(tf.shape(encoded_text)) * float('inf')
        if phoneme_max_duration is not None:
            for item in phoneme_max_duration.items():
                phon_idx = self.text_pipeline.tokenizer(item[0])[0]
                new_mask[np_text == phon_idx] = item[1]
        return tf.cast(tf.convert_to_tensor(new_mask), tf.float32)
    
    def _make_min_duration_mask(self, encoded_text, phoneme_min_duration):
        np_text = np.array(encoded_text)
        new_mask = np.zeros(tf.shape(encoded_text))
        if phoneme_min_duration is not None:
            for item in phoneme_min_duration.items():
                phon_idx = self.text_pipeline.tokenizer(item[0])[0]
                new_mask[np_text == phon_idx] = item[1]
        return tf.cast(tf.convert_to_tensor(new_mask), tf.float32)
    
    def build_model_weights(self) -> None:
        _ = self(tf.zeros((1, 1)), target_durations=None, target_pitch=None, training=False)
    
    def save_model(self, path: str):
        yaml = YAML()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self, 'text_pipeline'):
            save_list = ''.join(self.symbols)
            self.config.update({'alphabet': save_list})
        if hasattr(self, 'step'):
            self.config.update({'step': self.step})
        try:
            git_hash = subprocess.check_output(['git', 'describe', '--always']).strip().decode()
            self.config.update({'git_hash': git_hash})
        except Exception as e:
            print(f'WARNING: could not retrieve git hash. {e}')
        with open(path / 'config.yaml', 'w') as f:
            yaml.dump(dict(self.config), f)  # conversion necessary (is tf wrapper otherwise)
        # only needed when model was loaded from a checkpoint
        self.build_model_weights()
        self.save_weights(path / 'model_weights.hdf5')
    
    @classmethod
    def load_model(cls, path):
        yaml = YAML()
        path = Path(path)
        with open(path / 'config.yaml', 'r') as f:
            config = yaml.load(f)
        model = cls.from_config(config)
        try:
            git_hash = subprocess.check_output(['git', 'describe', '--always']).strip().decode()
            if 'git_hash' in config:
                if config['git_hash'] != git_hash:
                    print(f"WARNING: git_hash mismatch: {config['git_hash']}(config) vs {git_hash}(local).")
            else:
                print(f'WARNING: could not check git hash from config.')
        except Exception as e:
            print(f'WARNING: could not retrieve git hash. {e}')
        model.build_model_weights()
        model.load_weights(path / 'model_weights.hdf5')
        return model
    
    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        return cls(**config)
