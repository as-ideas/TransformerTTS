import sys

import tensorflow as tf

from model.transformer_utils import create_encoder_padding_mask, create_mel_padding_mask, create_look_ahead_mask
from utils.losses import weighted_sum_losses, masked_mean_absolute_error, new_scaled_crossentropy
from preprocessing.text import Pipeline
from model.layers import DecoderPrenet, Postnet, DurationPredictor, Expand, SelfAttentionBlocks, CrossAttentionBlocks, \
    CNNResNorm
from model.GST import GST


class AutoregressiveTransformer(tf.keras.models.Model):
    
    def __init__(self,
                 encoder_model_dimension: int,
                 decoder_model_dimension: int,
                 encoder_num_heads: list,
                 decoder_num_heads: list,
                 encoder_maximum_position_encoding: int,
                 decoder_maximum_position_encoding: int,
                 encoder_dense_blocks: int,
                 decoder_dense_blocks: int,
                 encoder_prenet_dimension: int,
                 decoder_prenet_dimension: int,
                 postnet_conv_filters: int,
                 postnet_conv_layers: int,
                 postnet_kernel_size: int,
                 dropout_rate: float,
                 mel_start_value: float,
                 mel_end_value: float,
                 mel_channels: int,
                 phoneme_language: str,
                 with_stress: bool,
                 num_style_heads: int,
                 num_style_tokens: int,
                 encoder_attention_conv_filters: int = None,
                 decoder_attention_conv_filters: int = None,
                 encoder_attention_conv_kernel: int = None,
                 decoder_attention_conv_kernel: int = None,
                 encoder_feed_forward_dimension: int = None,
                 decoder_feed_forward_dimension: int = None,
                 decoder_prenet_dropout=0.5,
                 max_r: int = 10,
                 debug=False,
                 **kwargs):
        super(AutoregressiveTransformer, self).__init__(**kwargs)
        self.start_vec = tf.ones((1, mel_channels), dtype=tf.float32) * mel_start_value
        self.end_vec = tf.ones((1, mel_channels), dtype=tf.float32) * mel_end_value
        self.stop_prob_index = 2
        self.max_r = max_r
        self.r = max_r
        self.mel_channels = mel_channels
        self.drop_n_heads = 0
        self.text_pipeline = Pipeline.default_pipeline(phoneme_language,
                                                       add_start_end=True,
                                                       with_stress=with_stress)
        self.encoder_prenet = tf.keras.layers.Embedding(self.text_pipeline.tokenizer.vocab_size,
                                                        encoder_prenet_dimension,
                                                        name='Embedding')
        self.encoder = SelfAttentionBlocks(model_dim=encoder_model_dimension,
                                           dropout_rate=dropout_rate,
                                           num_heads=encoder_num_heads,
                                           feed_forward_dimension=encoder_feed_forward_dimension,
                                           maximum_position_encoding=encoder_maximum_position_encoding,
                                           dense_blocks=encoder_dense_blocks,
                                           conv_filters=encoder_attention_conv_filters,
                                           kernel_size=encoder_attention_conv_kernel,
                                           conv_activation='relu',
                                           name='Encoder')
        self.gst = GST(model_size=encoder_model_dimension,
                       num_heads=num_style_heads,
                       token_num=num_style_tokens,
                       mel_channels=mel_channels)
        self.decoder_prenet = DecoderPrenet(model_dim=decoder_model_dimension,
                                            dense_hidden_units=decoder_prenet_dimension,
                                            dropout_rate=decoder_prenet_dropout,
                                            name='DecoderPrenet')
        self.decoder = CrossAttentionBlocks(model_dim=decoder_model_dimension,
                                            dropout_rate=dropout_rate,
                                            num_heads=decoder_num_heads,
                                            feed_forward_dimension=decoder_feed_forward_dimension,
                                            maximum_position_encoding=decoder_maximum_position_encoding,
                                            dense_blocks=decoder_dense_blocks,
                                            conv_filters=decoder_attention_conv_filters,
                                            conv_kernel=decoder_attention_conv_kernel,
                                            conv_activation='relu',
                                            conv_padding='causal',
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
        self.gst_signature = [
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
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
        self.forward_gst = self._apply_signature(self._forward_gst, self.gst_signature)
    
    def _call_encoder(self, inputs, training):
        padding_mask = create_encoder_padding_mask(inputs)
        enc_input = self.encoder_prenet(inputs)
        enc_output, attn_weights = self.encoder(enc_input,
                                                training=training,
                                                padding_mask=padding_mask,
                                                drop_n_heads=self.drop_n_heads)
        return enc_output, padding_mask, attn_weights
    
    def _call_decoder(self, encoder_output, targets, encoder_padding_mask, training):
        dec_target_padding_mask = create_mel_padding_mask(targets)
        look_ahead_mask = create_look_ahead_mask(tf.shape(targets)[1])
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        dec_input = self.decoder_prenet(targets)
        dec_output, attention_weights = self.decoder(inputs=dec_input,
                                                     enc_output=encoder_output,
                                                     training=training,
                                                     decoder_padding_mask=combined_mask,
                                                     encoder_padding_mask=encoder_padding_mask,
                                                     drop_n_heads=self.drop_n_heads,
                                                     reduction_factor=self.r)
        out_proj = self.final_proj_mel(dec_output)[:, :, :self.r * self.mel_channels]
        b = int(tf.shape(out_proj)[0])
        t = int(tf.shape(out_proj)[1])
        mel = tf.reshape(out_proj, (b, t * self.r, self.mel_channels))
        model_output = self.decoder_postnet(mel, training=training)
        model_output.update(
            {'decoder_attention': attention_weights, 'decoder_output': dec_output, 'linear': mel})
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
    
    def _forward_gst(self, reference_mel):
        # TODO: add mask
        style_emb, style_attn = self.gst(reference_mel, training=False)
        return style_emb, style_attn
    
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
    
    def _compile(self, stop_scaling, optimizer):
        self.loss_weights = [1., 1., 1.]
        self.compile(loss=[masked_mean_absolute_error,
                           new_scaled_crossentropy(index=2, scaling=stop_scaling),
                           masked_mean_absolute_error],
                     loss_weights=self.loss_weights,
                     optimizer=optimizer)
    
    def _set_r(self, r):
        if self.r == r:
            return
        self.r = r
        self._apply_all_signatures()
    
    def _set_heads(self, heads):
        if self.drop_n_heads == heads:
            return
        self.drop_n_heads = heads
        self._apply_all_signatures()
    
    def call(self, inputs, targets, training):
        encoder_output, padding_mask, encoder_attention = self._call_encoder(inputs, training)
        style_emb, style_attn = self.gst(targets, training=training)
        encoder_output = encoder_output + tf.expand_dims(style_emb, 1)
        model_out = self._call_decoder(encoder_output, targets, padding_mask, training)
        model_out.update({'encoder_attention': encoder_attention})
        model_out.update({'style_attention': style_attn})
        return model_out
    
    def predict(self, inp, ref_mel, max_length=1000, encode=True, verbose=True):
        if encode:
            inp = self.encode_text(inp)
        inp = tf.cast(tf.expand_dims(inp, 0), tf.int32)
        output = tf.cast(tf.expand_dims(self.start_vec, 0), tf.float32)
        output_concat = tf.cast(tf.expand_dims(self.start_vec, 0), tf.float32)
        out_dict = {}
        encoder_output, padding_mask, encoder_attention = self.forward_encoder(inp)
        if len(tf.shape(ref_mel)) < 3:
            ref_mel = tf.expand_dims(ref_mel, 0)
        style_emb, style_attn = self.forward_gst(ref_mel)
        encoder_output = encoder_output + tf.expand_dims(style_emb, 1)
        for i in range(int(max_length // self.r) + 1):
            model_out = self.forward_decoder(encoder_output, output, padding_mask)
            output = tf.concat([output, model_out['final_output'][:1, -1:, :]], axis=-2)
            output_concat = tf.concat([tf.cast(output_concat, tf.float32), model_out['final_output'][:1, -self.r:, :]],
                                      axis=-2)
            stop_pred = model_out['stop_prob'][:, -1]
            out_dict = {'mel': output_concat[0, 1:, :],
                        'decoder_attention': model_out['decoder_attention'],
                        'encoder_attention': encoder_attention,
                        'style_attention': style_attn}
            if verbose:
                sys.stdout.write(f'\rpred text mel: {i} stop out: {float(stop_pred[0, 2])}')
            if int(tf.argmax(stop_pred, axis=-1)) == self.stop_prob_index:
                if verbose:
                    print('Stopping')
                break
        return out_dict
    
    def set_constants(self, decoder_prenet_dropout: float = None, learning_rate: float = None,
                      reduction_factor: float = None, drop_n_heads: int = None):
        if decoder_prenet_dropout is not None:
            self.decoder_prenet.rate.assign(decoder_prenet_dropout)
        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)
        if reduction_factor is not None:
            self._set_r(reduction_factor)
        if drop_n_heads is not None:
            self._set_heads(drop_n_heads)
    
    def encode_text(self, text):
        return self.text_pipeline(text)


class ForwardTransformer(tf.keras.models.Model):
    def __init__(self,
                 encoder_model_dimension: int,
                 decoder_model_dimension: int,
                 dropout_rate: float,
                 decoder_num_heads: list,
                 encoder_num_heads: list,
                 encoder_maximum_position_encoding: int,
                 decoder_maximum_position_encoding: int,
                 postnet_conv_filters: int,
                 postnet_conv_layers: int,
                 postnet_kernel_size: int,
                 encoder_dense_blocks: int,
                 decoder_dense_blocks: int,
                 mel_channels: int,
                 phoneme_language: str,
                 with_stress: bool,
                 num_style_heads: int,
                 num_style_tokens: int,
                 encoder_attention_conv_filters: int = None,
                 decoder_attention_conv_filters: int = None,
                 encoder_attention_conv_kernel: int = None,
                 decoder_attention_conv_kernel: int = None,
                 encoder_feed_forward_dimension: int = None,
                 decoder_feed_forward_dimension: int = None,
                 debug=False,
                 decoder_prenet_dropout=0.,
                 **kwargs):
        super(ForwardTransformer, self).__init__(**kwargs)
        self.text_pipeline = Pipeline.default_pipeline(phoneme_language,
                                                       add_start_end=False,
                                                       with_stress=with_stress)
        self.drop_n_heads = 0
        self.mel_channels = mel_channels
        self.encoder_prenet = tf.keras.layers.Embedding(self.text_pipeline.tokenizer.vocab_size,
                                                        encoder_model_dimension,
                                                        name='Embedding')
        self.encoder = SelfAttentionBlocks(model_dim=encoder_model_dimension,
                                           dropout_rate=dropout_rate,
                                           num_heads=encoder_num_heads,
                                           feed_forward_dimension=encoder_feed_forward_dimension,
                                           maximum_position_encoding=encoder_maximum_position_encoding,
                                           dense_blocks=encoder_dense_blocks,
                                           conv_filters=encoder_attention_conv_filters,
                                           kernel_size=encoder_attention_conv_kernel,
                                           conv_activation='relu',
                                           name='Encoder')
        self.gst = GST(model_size=encoder_model_dimension,
                       num_heads=num_style_heads,
                       token_num=num_style_tokens,
                       mel_channels=mel_channels)
        self.dur_pred = DurationPredictor(model_dim=encoder_model_dimension,
                                          kernel_size=3,
                                          conv_padding='same',
                                          conv_activation='relu',
                                          conv_block_n=2,
                                          dense_activation='relu',
                                          name='dur_pred')
        self.expand = Expand(name='expand', model_dim=encoder_model_dimension)
        self.decoder_prenet = DecoderPrenet(model_dim=decoder_model_dimension,
                                            dense_hidden_units=decoder_feed_forward_dimension,
                                            dropout_rate=decoder_prenet_dropout,
                                            name='DecoderPrenet')
        self.decoder = SelfAttentionBlocks(model_dim=decoder_model_dimension,
                                           dropout_rate=dropout_rate,
                                           num_heads=decoder_num_heads,
                                           feed_forward_dimension=decoder_feed_forward_dimension,
                                           maximum_position_encoding=decoder_maximum_position_encoding,
                                           dense_blocks=decoder_dense_blocks,
                                           conv_filters=decoder_attention_conv_filters,
                                           kernel_size=decoder_attention_conv_kernel,
                                           conv_activation='relu',
                                           name='Decoder')
        self.out = tf.keras.layers.Dense(mel_channels)
        self.decoder_postnet = CNNResNorm(out_size=mel_channels,
                                          kernel_size=postnet_kernel_size,
                                          padding='same',
                                          inner_activation='tanh',
                                          last_activation='linear',
                                          hidden_size=postnet_conv_filters,
                                          n_layers=postnet_conv_layers,
                                          normalization='batch',
                                          name='Postnet')
        self.training_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32)
        ]
        self.forward_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),  # reference mel
            tf.TensorSpec(shape=(), dtype=tf.float32),
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
    
    def _set_heads(self, heads):
        if self.drop_n_heads == heads:
            return
        self.drop_n_heads = heads
        self._apply_all_signatures()
    
    def _train_step(self, input_sequence, target_sequence, target_durations):
        target_durations = tf.expand_dims(target_durations, -1)
        mel_len = int(tf.shape(target_sequence)[1])
        with tf.GradientTape() as tape:
            model_out = self.__call__(input_sequence,
                                      target_durations=target_durations,
                                      reference_mel=target_sequence,
                                      training=True)
            loss, loss_vals = weighted_sum_losses((target_sequence,
                                                   target_durations),
                                                  (model_out['mel'][:, :mel_len, :],
                                                   model_out['duration']),
                                                  self.loss,
                                                  self.loss_weights)
        model_out.update({'loss': loss})
        model_out.update({'losses': {'mel': loss_vals[0], 'duration': loss_vals[1]}})
        gradients = tape.gradient(model_out['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out
    
    def _compile(self, optimizer):
        self.loss_weights = [3., 1.]
        self.compile(loss=[masked_mean_absolute_error,
                           masked_mean_absolute_error],
                     loss_weights=self.loss_weights,
                     optimizer=optimizer)
    
    def _val_step(self, input_sequence, target_sequence, target_durations):
        target_durations = tf.expand_dims(target_durations, -1)
        mel_len = int(tf.shape(target_sequence)[1])
        model_out = self.__call__(input_sequence,
                                  target_durations=target_durations,
                                  reference_mel=target_sequence,
                                  training=False)
        loss, loss_vals = weighted_sum_losses((target_sequence,
                                               target_durations),
                                              (model_out['mel'][:, :mel_len, :],
                                               model_out['duration']),
                                              self.loss,
                                              self.loss_weights)
        model_out.update({'loss': loss})
        model_out.update({'losses': {'mel': loss_vals[0], 'duration': loss_vals[1]}})
        return model_out
    
    def _forward(self, input_sequence, reference_mel, durations_scalar):
        return self.__call__(input_sequence, target_durations=None, reference_mel=reference_mel, training=False,
                             durations_scalar=durations_scalar)
    
    @property
    def step(self):
        return int(self.optimizer.iterations)
    
    def call(self, x, target_durations, reference_mel, training, durations_scalar=1.):
        padding_mask = create_encoder_padding_mask(x)
        x = self.encoder_prenet(x)
        x, encoder_attention = self.encoder(x, training=training, padding_mask=padding_mask,
                                            drop_n_heads=self.drop_n_heads)
        style_emb, style_attn = self.gst(reference_mel, training=training)  # TODO: add mask
        x = x + tf.expand_dims(style_emb, 1)
        durations = self.dur_pred(x, training=training) * durations_scalar
        durations = (1. - tf.reshape(padding_mask, tf.shape(durations))) * durations
        if target_durations is not None:
            mels = self.expand(x, target_durations)
        else:
            mels = self.expand(x, durations)
        expanded_mask = create_mel_padding_mask(mels)
        mels = self.decoder_prenet(mels)
        mels, decoder_attention = self.decoder(mels, training=training, padding_mask=expanded_mask,
                                               drop_n_heads=self.drop_n_heads, reduction_factor=1)
        mels = self.out(mels)
        mels = self.decoder_postnet(mels, training=training)
        model_out = {'mel': mels,
                     'duration': durations,
                     'expanded_mask': expanded_mask,
                     'encoder_attention': encoder_attention,
                     'decoder_attention': decoder_attention,
                     'style_attention': style_attn}
        return model_out
    
    def set_constants(self, decoder_prenet_dropout: float = None, learning_rate: float = None,
                      drop_n_heads: int = None, **kwargs):
        if decoder_prenet_dropout is not None:
            self.decoder_prenet.rate.assign(decoder_prenet_dropout)
        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)
        if drop_n_heads is not None:
            self._set_heads(drop_n_heads)
    
    def encode_text(self, text):
        return self.text_pipeline(text)
    
    def predict(self, inp, reference_mel, encode=True, speed_regulator=1.):
        if encode:
            inp = self.encode_text(inp)
            inp = tf.cast(tf.expand_dims(inp, 0), tf.int32)
        duration_scalar = tf.cast(1. / speed_regulator, tf.float32)
        out = self.forward(inp, reference_mel=reference_mel, durations_scalar=duration_scalar)
        out['mel'] = tf.squeeze(out['mel'])
        return out


if __name__ == '__main__':
    bs = 3
    input_text = tf.random.uniform((bs, 60))
    target_mel = tf.random.uniform((bs, 100, 80))
    model = AutoregressiveTransformer(
        decoder_model_dimension=256,
        encoder_model_dimension=512,
        decoder_num_heads=[4, 4, 4, 4],
        encoder_num_heads=[4, 4, 4, 4],
        encoder_maximum_position_encoding=1000,
        decoder_maximum_position_encoding=10000,
        encoder_dense_blocks=4,
        decoder_dense_blocks=4,
        encoder_prenet_dimension=512,
        decoder_prenet_dimension=256,
        postnet_conv_filters=256,
        postnet_conv_layers=2,
        postnet_kernel_size=5,
        dropout_rate=0.1,
        mel_start_value=-.5,
        mel_end_value=.5,
        mel_channels=80,
        phoneme_language='en',
        with_stress=True,
        num_style_heads=8,
        num_style_tokens=10,
        encoder_feed_forward_dimension=1024,
        decoder_feed_forward_dimension=1024,
        debug=True, )
    model_out = model.forward(input_text, target_mel)
    pass
