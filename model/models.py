import tensorflow as tf

from model.transformer_utils import create_text_padding_mask, create_mel_padding_mask, create_look_ahead_mask
from utils.losses import weighted_sum_losses


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


class TextTransformer(Transformer):
    
    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 decoder_postnet,
                 encoder,
                 decoder,
                 tokenizer,
                 debug=False):
        super(TextTransformer, self).__init__(encoder_prenet,
                                              decoder_prenet,
                                              encoder,
                                              decoder,
                                              decoder_postnet)
        self.tokenizer = tokenizer
        self._check_tokenizer()
        input_signature = [
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32)]
        if not debug:
            self.train_step = tf.function(input_signature=input_signature)(self._train_step)
            self.val_step = tf.function(input_signature=input_signature)(self._val_step)
        else:
            self.train_step = self._train_step
            self.val_step = self._val_step

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

    def predict(self, inputs, max_length=40, encode=False):
        if encode:
            inputs = self.tokenizer.encode(inputs)
        encoder_input = tf.expand_dims(inputs, 0)
        decoder_input = [self.tokenizer.start_token_index]
        output = tf.expand_dims(decoder_input, 0)
        out_dict = {}
        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(encoder_input, output)
            model_out = self.__call__(inputs=encoder_input,
                                      targets=output,
                                      training=False,
                                      enc_padding_mask=enc_padding_mask,
                                      look_ahead_mask=combined_mask,
                                      dec_padding_mask=dec_padding_mask)
            predictions = model_out['final_output'][:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            output = tf.concat([output, predicted_id], axis=-1)
            out_dict = {'output': tf.squeeze(output, axis=0),
                        'attention_weights': model_out['attention_weights'],
                        'logits': predictions}
            if predicted_id == self.tokenizer.end_token_index:
                break
        return out_dict
    
    def create_masks(self, inp, tar_inp):
        enc_padding_mask = create_text_padding_mask(inp)
        dec_padding_mask = create_text_padding_mask(inp)
        dec_target_padding_mask = create_text_padding_mask(tar_inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask
    
    def _train_step(self, inp, tar):
        model_out, tape = self._forward_pass(inp, tar, training=True)
        gradients = tape.gradient(model_out['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out

    def _val_step(self, inp, tar):
        model_out, _ = self._forward_pass(inp, tar, training=False)
        return model_out

    def _forward_pass(self, inp, tar, training):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            model_out = self.__call__(inputs=inp,
                                      targets=tar_inp,
                                      training=training,
                                      enc_padding_mask=enc_padding_mask,
                                      look_ahead_mask=combined_mask,
                                      dec_padding_mask=dec_padding_mask)
            loss = self.loss(tar_real, model_out['final_output'])
        model_out.update({'loss': loss})
        return model_out, tape

    def _check_tokenizer(self):
        for attribute in ['start_token_index', 'end_token_index', 'vocab_size']:
            assert hasattr(self.tokenizer, attribute), f'Tokenizer is missing {attribute}.'


class MelTransformer(Transformer):
    
    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 encoder,
                 decoder,
                 decoder_postnet,
                 start_vec_value=-3,
                 end_vec_value=1,
                 debug=False):
        super(MelTransformer, self).__init__(encoder_prenet,
                                             decoder_prenet,
                                             encoder,
                                             decoder,
                                             decoder_postnet)
        self.start_vec = tf.ones((1, decoder_postnet.mel_channels), dtype=tf.float32) * start_vec_value
        self.end_vec = tf.ones((1, decoder_postnet.mel_channels), dtype=tf.float32) * end_vec_value
        self.stop_prob_index = 2
        input_signature = [
            tf.TensorSpec(shape=(None, None, decoder_postnet.mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, decoder_postnet.mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None), dtype=tf.float32)
        ]
        if not debug:
            self.train_step = tf.function(input_signature=input_signature)(self._train_step)
            self.val_step = tf.function(input_signature=input_signature)(self._val_step)
        else:
            self.train_step = self._train_step
            self.val_step = self._val_step

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
        model_output = self.decoder_postnet(inputs=dec_output, training=training)
        model_output.update({'attention_weights': attention_weights, 'decoder_output': dec_output})
        return model_output

    def predict(self, inputs, max_length=50, decoder_prenet_dropout=0.5):
        inputs = tf.expand_dims(inputs, 0)
        output = tf.expand_dims(self.start_vec, 0)
        out_dict = {}
        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inputs, output)
            model_out = self.__call__(inputs=inputs,
                                      targets=output,
                                      training=False,
                                      enc_padding_mask=enc_padding_mask,
                                      look_ahead_mask=combined_mask,
                                      dec_padding_mask=dec_padding_mask,
                                      decoder_prenet_dropout=decoder_prenet_dropout)
            output = tf.concat([output, model_out['final_output'][:1, -1:, :]], axis=-2)
            stop_pred = model_out['stop_prob'][:, -1]
            out_dict = {'mel': output[0, 1:, :], 'attention_weights': model_out['attention_weights']}
            if int(tf.argmax(stop_pred, axis=-1)) == self.stop_prob_index:
                break
        return out_dict
    
    def create_masks(self, inp, tar_inp):
        enc_padding_mask = create_mel_padding_mask(inp)
        dec_padding_mask = create_mel_padding_mask(inp)
        dec_target_padding_mask = create_mel_padding_mask(tar_inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask
    
    def _train_step(self, inp, tar, stop_prob, decoder_prenet_dropout):
        model_out, tape = self._forward_pass(inp,
                                             tar,
                                             stop_prob,
                                             decoder_prenet_dropout,
                                             training=True)
        gradients = tape.gradient(model_out['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out

    def _val_step(self, inp, tar, stop_prob, decoder_prenet_dropout):
        model_out, _ = self._forward_pass(inp,
                                             tar,
                                             stop_prob,
                                             decoder_prenet_dropout,
                                             training=False)
        return model_out

    def _forward_pass(self, inp, tar, stop_prob, decoder_prenet_dropout, training):
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]
        tar_stop_prob = stop_prob[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            model_out = self.__call__(inputs=inp,
                                      targets=tar_inp,
                                      training=training,
                                      enc_padding_mask=enc_padding_mask,
                                      look_ahead_mask=combined_mask,
                                      dec_padding_mask=dec_padding_mask,
                                      decoder_prenet_dropout=decoder_prenet_dropout)
            loss, loss_vals = weighted_sum_losses(
                (tar_real, tar_stop_prob, tar_real),
                (model_out['final_output'], model_out['stop_prob'], model_out['mel_linear']),
                self.loss,
                self.loss_weights,
            )
        model_out.update({'loss': loss})
        model_out.update({'losses': {'output': loss_vals[0], 'stop_prob': loss_vals[1], 'mel_linear': loss_vals[2]}})
        return model_out, tape


class MelTextTransformer(Transformer):
    
    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 decoder_postnet,
                 encoder,
                 decoder,
                 tokenizer,
                 mel_channels,
                 start_vec_value=-3,
                 end_vec_value=1,
                 debug=False):
        super(MelTextTransformer, self).__init__(encoder_prenet,
                                                 decoder_prenet,
                                                 encoder,
                                                 decoder,
                                                 decoder_postnet)
        self.tokenizer = tokenizer
        self._check_tokenizer()
        self.start_vec = tf.ones((1, mel_channels), dtype=tf.float32) * start_vec_value
        self.end_vec = tf.ones((1, mel_channels), dtype=tf.float32) * end_vec_value
        input_signature=[
                    tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                ]
        if not debug:
            self.train_step = tf.function(input_signature=input_signature)(self._train_step)
            self.val_step = tf.function(input_signature=input_signature)(self._val_step)
        else:
            self.train_step = self._train_step
            self.val_step = self._val_step

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

    def predict(self, inputs, max_length=100):
        encoder_input = tf.expand_dims(inputs, 0)
        decoder_input = [self.tokenizer.start_token_index]
        output = tf.expand_dims(decoder_input, 0)
        out_dict = {}
        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(encoder_input, output)
            model_out = self.__call__(inputs=encoder_input,
                                      targets=output,
                                      training=False,
                                      enc_padding_mask=enc_padding_mask,
                                      look_ahead_mask=combined_mask,
                                      dec_padding_mask=dec_padding_mask)
            predictions = model_out['final_output'][:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            output = tf.concat([output, predicted_id], axis=-1)
            out_dict = {'output': tf.squeeze(output, axis=0),
                        'attention_weights': model_out['attention_weights'],
                        'logits': predictions}
            if predicted_id == self.tokenizer.end_token_index:
                break
        return out_dict
    
    def create_masks(self, inp, tar_inp):
        enc_padding_mask = create_mel_padding_mask(inp)
        dec_padding_mask = create_mel_padding_mask(inp)
        dec_target_padding_mask = create_text_padding_mask(tar_inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask

    def _forward_pass(self, inp, tar, training):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            model_out = self.__call__(inputs=inp,
                                      targets=tar_inp,
                                      training=training,
                                      enc_padding_mask=enc_padding_mask,
                                      look_ahead_mask=combined_mask,
                                      dec_padding_mask=dec_padding_mask)
            loss = self.loss(tar_real, model_out['final_output'])
        model_out.update({'loss': loss})
        return model_out, tape

    def _train_step(self, inp, tar):
        model_out, tape = self._forward_pass(inp, tar, training=True)
        gradients = tape.gradient(model_out['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out

    def _val_step(self, inp, tar):
        model_out, _ = self._forward_pass(inp, tar, training=False)
        return model_out

    def _check_tokenizer(self):
        for attribute in ['start_token_index', 'end_token_index', 'vocab_size']:
            assert hasattr(self.tokenizer, attribute), f'Tokenizer is missing {attribute}.'


class TextMelTransformer(Transformer):
    
    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 encoder,
                 decoder,
                 decoder_postnet,
                 tokenizer,
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
        self.train_step = self._forward_pass
        input_signature = [
                    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                    tf.TensorSpec(shape=(None, None, decoder_postnet.mel_channels), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                    tf.TensorSpec(shape=(None), dtype=tf.float32)
                ]
        if not debug:
            self.train_step = tf.function(input_signature=input_signature)(self._train_step)
            self.val_step = tf.function(input_signature=input_signature)(self._val_step)
        else:
            self.train_step = self._train_step
            self.val_step = self._val_step

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
        model_output = self.decoder_postnet(inputs=dec_output, training=training)
        model_output.update({'attention_weights': attention_weights, 'decoder_output': dec_output})
        return model_output

    def predict(self, inp, max_length=50, decoder_prenet_dropout=0.5):
        inp = tf.expand_dims(inp, 0)
        output = tf.expand_dims(self.start_vec, 0)
        out_dict = {}
        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, output)
            model_out = self.__call__(inputs=inp,
                                      targets=output,
                                      training=False,
                                      enc_padding_mask=enc_padding_mask,
                                      look_ahead_mask=combined_mask,
                                      dec_padding_mask=dec_padding_mask,
                                      decoder_prenet_dropout=decoder_prenet_dropout)
            output = tf.concat([tf.cast(output, tf.float32), model_out['final_output'][:1, -1:, :]], axis=-2)
            stop_pred = model_out['stop_prob'][:, -1]
            out_dict = {'mel': output[0, 1:, :], 'attention_weights': model_out['attention_weights']}
            if int(tf.argmax(stop_pred, axis=-1)) == self.stop_prob_index:
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

    def _forward_pass(self, inp, tar, stop_prob, decoder_prenet_dropout, training):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        tar_stop_prob = stop_prob[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            model_out = self.__call__(inputs=inp,
                                      targets=tar_inp,
                                      training=training,
                                      enc_padding_mask=enc_padding_mask,
                                      look_ahead_mask=combined_mask,
                                      dec_padding_mask=dec_padding_mask,
                                      decoder_prenet_dropout=decoder_prenet_dropout)
            loss, loss_vals = weighted_sum_losses((tar_real, tar_stop_prob, tar_real),
                                                  (model_out['final_output'],
                                                   model_out['stop_prob'],
                                                   model_out['mel_linear']),
                                                  self.loss,
                                                  self.loss_weights)
        model_out.update({'loss': loss})
        model_out.update({'losses': {'output': loss_vals[0], 'stop_prob': loss_vals[1], 'mel_linear': loss_vals[2]}})
        return model_out, tape

    def _train_step(self, inp, tar, stop_prob, decoder_prenet_dropout):
        model_out, tape = self._forward_pass(inp, tar, stop_prob, decoder_prenet_dropout, training=True)
        gradients = tape.gradient(model_out['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return model_out

    def _val_step(self, inp, tar, stop_prob, decoder_prenet_dropout):
        model_out, _ = self._forward_pass(inp, tar, stop_prob, decoder_prenet_dropout, training=False)
        return model_out

    def _check_tokenizer(self):
        for attribute in ['start_token_index', 'end_token_index', 'vocab_size']:
            assert hasattr(self.tokenizer, attribute), f'Tokenizer is missing {attribute}.'
