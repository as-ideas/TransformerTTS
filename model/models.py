import sys

import tensorflow as tf

from model.transformer_utils import create_encoder_padding_mask, create_mel_padding_mask, create_look_ahead_mask
from utils.losses import weighted_sum_losses
from model.layers import DecoderPrenet, Postnet, Decoder, Encoder
from utils.losses import masked_mean_absolute_error, new_scaled_crossentropy
from preprocessing.tokenizer import Tokenizer
from preprocessing.text_processing import _phonemes, Phonemizer, _punctuations


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
                 max_r: int = 10,
                 mel_start_value: int = -3,
                 mel_end_value: int = 1,
                 phoneme_language: str = 'en',
                 debug=False,
                 **kwargs):
        super(AutoregressiveTransformer, self).__init__(**kwargs)
        self.start_vec = tf.ones((1, mel_channels), dtype=tf.float32) * mel_start_value
        self.end_vec = tf.ones((1, mel_channels), dtype=tf.float32) * mel_end_value
        self.stop_prob_index = 2
        self.max_r = max_r
        self.r = max_r
        self.mel_channels = mel_channels
        
        self.tokenizer = Tokenizer(sorted(list(_phonemes) + list(_punctuations)))
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
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None), dtype=tf.float32)
        ]
        self.forward_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None), dtype=tf.float32)
        ]
        self.debug = debug
        self.forward = self.__apply_signature(self._forward, self.forward_input_signature)
        self.train_step = self.__apply_signature(self._train_step, self.training_input_signature)
        self.val_step = self.__apply_signature(self._val_step, self.training_input_signature)
    
    def __apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)
    
    def call(self,
             inputs,
             targets,
             training,
             enc_padding_mask,
             look_ahead_mask,
             dec_padding_mask,
             decoder_prenet_dropout):
        enc_input = self.encoder_prenet(inputs)
        enc_output = self.encoder(inputs=enc_input,
                                  training=training,
                                  mask=enc_padding_mask)
        dec_input = self.decoder_prenet(targets, training=training, dropout_rate=decoder_prenet_dropout)
        dec_output, attention_weights = self.decoder(inputs=dec_input,
                                                     enc_output=enc_output,
                                                     training=training,
                                                     look_ahead_mask=look_ahead_mask,
                                                     padding_mask=dec_padding_mask)
        out_proj = self.final_proj_mel(dec_output)[:, :, :self.r * self.mel_channels]
        b = int(tf.shape(out_proj)[0])
        t = int(tf.shape(out_proj)[1])
        mel = tf.reshape(out_proj, (b, t * self.r, self.mel_channels))
        model_output = self.decoder_postnet(inputs=mel, training=training)
        model_output.update(
            {'attention_weights': attention_weights, 'decoder_output': dec_output, 'out_proj': out_proj})
        return model_output
    
    def predict(self, inp, max_length=50, decoder_prenet_dropout=0.5, encode=False, verbose=True):
        if encode:
            inp = self.encode_text(inp)
        inp = tf.cast(tf.expand_dims(inp, 0), tf.int32)
        output = tf.cast(tf.expand_dims(self.start_vec, 0), tf.float32)
        output_concat = tf.cast(tf.expand_dims(self.start_vec, 0), tf.float32)
        out_dict = {}
        for i in range(int(max_length // self.r) + 1):
            model_out = self.forward(inp=inp,
                                     output=output,
                                     decoder_prenet_dropout=decoder_prenet_dropout)
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
    
    def create_masks(self, inp, tar_inp):
        enc_padding_mask = create_encoder_padding_mask(inp)
        dec_padding_mask = create_encoder_padding_mask(inp)
        dec_target_padding_mask = create_mel_padding_mask(tar_inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask
    
    def set_r(self, r):
        if self.r == r:
            return
        self.r = r
        self.forward = self.__apply_signature(self._forward, self.forward_input_signature)
        self.train_step = self.__apply_signature(self._train_step, self.training_input_signature)
        self.val_step = self.__apply_signature(self._val_step, self.training_input_signature)
    
    def _forward(self, inp, output, decoder_prenet_dropout):
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, output)
        model_out = self.__call__(inputs=inp,
                                  targets=output,
                                  training=False,
                                  enc_padding_mask=enc_padding_mask,
                                  look_ahead_mask=combined_mask,
                                  dec_padding_mask=dec_padding_mask,
                                  decoder_prenet_dropout=decoder_prenet_dropout)
        return model_out
    
    def _train_forward(self, inp, tar, stop_prob, decoder_prenet_dropout, training):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        tar_stop_prob = stop_prob[:, 1:]
        
        mel_len = int(tf.shape(tar_inp)[1])
        tar_mel = tar_inp[:, 0::self.r, :]
        
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_mel)
        with tf.GradientTape() as tape:
            model_out = self.__call__(inputs=inp,
                                      targets=tar_mel,
                                      training=training,
                                      enc_padding_mask=enc_padding_mask,
                                      look_ahead_mask=combined_mask,
                                      dec_padding_mask=dec_padding_mask,
                                      decoder_prenet_dropout=decoder_prenet_dropout)
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
    
    def _train_step(self, inp, tar, stop_prob, decoder_prenet_dropout):
        model_out, tape = self._train_forward(inp, tar, stop_prob, decoder_prenet_dropout, training=True)
        gradients = tape.gradient(model_out['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out
    
    def _val_step(self, inp, tar, stop_prob, decoder_prenet_dropout):
        model_out, _ = self._train_forward(inp, tar, stop_prob, decoder_prenet_dropout, training=False)
        return model_out
    
    @property
    def step(self):
        return int(self.optimizer.iterations)
    
    def set_learning_rates(self, new_lr):
        self.optimizer.lr.assign(new_lr)
    
    def _compile(self, stop_scaling, optimizer):
        self.compile(loss=[masked_mean_absolute_error,
                           new_scaled_crossentropy(index=2, scaling=stop_scaling),
                           masked_mean_absolute_error],
                     loss_weights=[1., 1., 1.],
                     optimizer=optimizer)
    
    def build_graph(self, r: int):
        self.set_r(r)
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
        self.set_r(r)
        return ckpt, manager
    
    def encode_text(self, text):
        phons = self.phonemizer.encode(text, clean=True)
        return self.tokenizer.encode(phons, add_start_end=True)
