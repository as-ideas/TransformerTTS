import sys

import tensorflow as tf

from model.transformer_utils import create_text_padding_mask, create_mel_padding_mask, create_look_ahead_mask
from utils.losses import weighted_sum_losses


def train_signature(f):
    def apply_func(*args, **kwargs):
        if args[0].debug:
            return f(*args, **kwargs)
        else:
            return tf.function(input_signature=args[0].train_input_signature)(f(*args, **kwargs))
    
    return apply_func


def forward_signature(f):
    def apply_func(*args, **kwargs):
        if args[0].debug:
            return f(*args, **kwargs)
        else:
            return tf.function(input_signature=args[0].forward_input_signature)(f(*args, **kwargs))
    
    return apply_func


class Transformer(tf.keras.Model):
    
    def __init__(
            self, encoder_prenet, decoder_prenet, encoder, decoder, decoder_postnet):
        super(Transformer, self).__init__()
        self.encoder_prenet = encoder_prenet
        self.decoder_prenet = decoder_prenet
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_postnet = decoder_postnet
    
    def call(self,
             inputs,
             targets,
             training,
             enc_padding_mask,
             look_ahead_mask,
             dec_padding_mask):
        enc_input = self.encoder_prenet(inputs)
        enc_output = self.encoder(inputs=enc_input,
                                  training=training,
                                  mask=enc_padding_mask)
        dec_input = self.decoder_prenet(targets, training=training)
        dec_output, attention_weights = self.decoder(inputs=dec_input,
                                                     enc_output=enc_output,
                                                     training=training,
                                                     look_ahead_mask=look_ahead_mask,
                                                     padding_mask=dec_padding_mask)
        model_output = self.decoder_postnet(inputs=dec_output, training=training)
        model_output.update({'attention_weights': attention_weights, 'decoder_output': dec_output})
        return model_output
    
    def predict(self, inputs, max_length=None):
        raise NotImplementedError()
    
    def create_masks(self, inputs, tar_inputs):
        raise NotImplementedError()


class TextMelTransformer(Transformer):
    
    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 encoder,
                 decoder,
                 decoder_postnet,
                 tokenizer,
                 r=10,
                 start_vec_value=-3,
                 end_vec_value=1,
                 debug=False):
        super(TextMelTransformer, self).__init__(encoder_prenet,
                                                 decoder_prenet,
                                                 encoder,
                                                 decoder,
                                                 decoder_postnet)
        self.start_vec = tf.ones((1, decoder_postnet.mel_channels), dtype=tf.float32) * start_vec_value
        self.end_vec = tf.ones((1, decoder_postnet.mel_channels), dtype=tf.float32) * end_vec_value
        self.tokenizer = tokenizer
        self._check_tokenizer()
        self.stop_prob_index = 2
        self.max_r = 10
        self.r = r
        self.mel_channels = 80
        self.final_proj_mel = tf.keras.layers.Dense(self.mel_channels * self.max_r)
        self.training_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, decoder_postnet.mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None), dtype=tf.float32)
        ]
        self.forward_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, decoder_postnet.mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None), dtype=tf.float32)
        ]
        self.debug = debug
    
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
        mel = tf.reshape(out_proj, (b, t*self.r, self.mel_channels))
        model_output = self.decoder_postnet(inputs=mel, training=training)
        model_output.update({'attention_weights': attention_weights, 'decoder_output': dec_output})
        return model_output
    
    def predict(self, inp, max_length=50, decoder_prenet_dropout=0.5, verbose=True):
        inp = tf.expand_dims(inp, 0)
        inp = tf.cast(inp, tf.int32)
        output = tf.expand_dims(self.start_vec, 0)
        output = tf.cast(output, tf.float32)
        output_concat = tf.expand_dims(self.start_vec, 0)
        output_concat = tf.cast(output_concat, tf.float32)
        out_dict = {}
        for i in range(int(max_length // self.r) + 1):
            model_out = self.forward(inp=inp,
                                     output=output,
                                     stop_prob=tf.zeros((1, 1), dtype=tf.int32),
                                     decoder_prenet_dropout=decoder_prenet_dropout)
            output = tf.concat([output, model_out['final_output'][:1, -1:, :]], axis=-2)
            output_concat = tf.concat([tf.cast(output_concat, tf.float32), model_out['final_output'][:1, -self.r:, :]],axis=-2)
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
        enc_padding_mask = create_text_padding_mask(inp)
        dec_padding_mask = create_text_padding_mask(inp)
        dec_target_padding_mask = create_mel_padding_mask(tar_inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask

    def set_r(self, r, debug=False):
        if self.r == r:
            return
        print(f'Text mel setting reduction factor from {self.r} to {r}')
        self.r = r
        input_signature = [
                    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                    tf.TensorSpec(shape=(None, None, self.decoder_postnet.mel_channels), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                    tf.TensorSpec(shape=(None), dtype=tf.float32)
                ]
        if not debug:
            self.train_step = tf.function(input_signature=input_signature)(self._train_step)
            self.val_step = tf.function(input_signature=input_signature)(self._val_step)
            self.forward = tf.function(input_signature=input_signature)(self._forward)
        else:
            self.train_step = self._train_step
            self.val_step = self._val_step
            self.forward = self._forward

    def _forward(self, inp, output, stop_prob, decoder_prenet_dropout):
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
        return model_out, tape
    
    @train_signature
    def train_step(self, inp, tar, stop_prob, decoder_prenet_dropout):
        model_out, tape = self._train_forward(inp, tar, stop_prob, decoder_prenet_dropout, training=True)
        gradients = tape.gradient(model_out['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out
    
    @train_signature
    def val_step(self, inp, tar, stop_prob, decoder_prenet_dropout):
        model_out, _ = self._train_forward(inp, tar, stop_prob, decoder_prenet_dropout, training=False)
        return model_out
    
    def _check_tokenizer(self):
        for attribute in ['start_token_index', 'end_token_index', 'vocab_size']:
            assert hasattr(self.tokenizer, attribute), f'Tokenizer is missing {attribute}.'
