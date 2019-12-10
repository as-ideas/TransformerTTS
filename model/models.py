import tensorflow as tf
from model.transformer_utils import weighted_sum_losses


class Transformer(tf.keras.Model):
    def __init__(
            self, encoder_prenet, decoder_prenet, encoder, decoder, decoder_postnet
    ):
        super(Transformer, self).__init__()
        self.encoder_prenet = encoder_prenet
        self.decoder_prenet = decoder_prenet
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_postnet = decoder_postnet
    
    def call(
            self,
            input_vecs,
            target_vecs,
            training,
            enc_padding_mask,
            look_ahead_mask,
            dec_padding_mask,
    ):
        enc_input = self.encoder_prenet(input_vecs)
        enc_output = self.encoder(inputs=enc_input, training=training, mask=enc_padding_mask)
        dec_input = self.decoder_prenet(target_vecs, training=training)
        dec_output, attention_weights = self.decoder(inputs=dec_input,
                                                     enc_output=enc_output,
                                                     training=training,
                                                     look_ahead_mask=look_ahead_mask,
                                                     padding_mask=dec_padding_mask)
        model_output = self.decoder_postnet(inputs=dec_output, training=training
                                            )
        model_output.update({'attention_weights': attention_weights,
                             'decoder_output': dec_output})
        return model_output
    
    def create_input_padding_mask(self, seq):
        if self.encoder_seq_dim == -2:
            seq = tf.reduce_sum(seq, axis=-1)
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, y, x)
    
    def create_target_padding_mask(self, seq):
        if self.decoder_seq_dim == -2:
            seq = tf.reduce_sum(seq, axis=-1)
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, y, x)
    
    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    
    def create_masks(self, inp, tar):
        enc_padding_mask = self.create_input_padding_mask(inp)
        dec_padding_mask = self.create_input_padding_mask(inp)
        dec_target_padding_mask = self.create_target_padding_mask(tar)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[self.decoder_seq_dim])
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask
    
    def _train_step_(self, inp, targets):
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, targets['input'])
        with tf.GradientTape() as tape:
            model_out = self.__call__(inp, targets['input'], True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = self._eval_loss(targets, model_out)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        model_out.update({'loss': loss})
        return model_out


class TextTransformer(Transformer):
    
    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 decoder_postnet,
                 encoder,
                 decoder,
                 start_token_index,
                 end_token_index):
        super(TextTransformer, self).__init__(encoder_prenet, decoder_prenet, encoder, decoder, decoder_postnet)
        self.start_token_index = start_token_index
        self.end_token_index = end_token_index
        self.decoder_seq_dim = -1
        self.encoder_seq_dim = -1
        self.train_step = tf.function(input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
        )(self._train_step)
    
    def _eval_loss(self, targets: dict, model_out: dict):
        loss = self.loss(targets['real'], model_out['final_output'])
        return loss
    
    def _train_step(self, inp, tar):
        targets = {'input': tar[:, :-1],
                   'real': tar[:, 1:]}
        return self._train_step_(inp, targets)
    
    def predict(self, encoded_inp_sentence, MAX_LENGTH=40):
        encoder_input = tf.expand_dims(encoded_inp_sentence, 0)
        decoder_input = [self.start_token_index]
        output = tf.expand_dims(decoder_input, 0)
        out_dict = {}
        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(encoder_input, output)
            model_out = self.__call__(
                encoder_input,
                target_vecs=output,
                training=False,
                enc_padding_mask=enc_padding_mask,
                look_ahead_mask=combined_mask,
                dec_padding_mask=dec_padding_mask,
            )
            predictions = model_out['final_output'][:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            output = tf.concat([output, predicted_id], axis=-1)
            out_dict = {'output': tf.squeeze(output, axis=0), 'attn_weights': model_out['attention_weights'],
                        'logits': predictions}
            if predicted_id == self.end_token_index:
                break
        return out_dict


class MelTransformer(Transformer):
    
    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 encoder,
                 decoder,
                 decoder_postnet,
                 start_vec,
                 stop_prob_index):
        super(MelTransformer, self).__init__(encoder_prenet, decoder_prenet, encoder, decoder, decoder_postnet)
        self.start_vec = tf.cast(start_vec, tf.float32)
        self.stop_prob_index = stop_prob_index
        self.train_step = tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, None, decoder_postnet.mel_channels), dtype=tf.float64),
                tf.TensorSpec(shape=(None, None, decoder_postnet.mel_channels), dtype=tf.float64),
                tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            ]
        )(self._train_step)
        self.decoder_seq_dim = -2
        self.encoder_seq_dim = -2
    
    def _eval_loss(self, targets: dict, model_out: dict):
        loss, _ = weighted_sum_losses((targets['real'], targets['tar_stop_prob'], targets['real']),
                                      (model_out['final_output'], model_out['stop_prob'], model_out['mel_linear']),
                                      self.loss,
                                      self.loss_weights)
        return loss
    
    def _train_step(self, inp, tar, stop_prob):
        targets = {'input': tar[:, :-1, :],
                   'real': tar[:, 1:, :],
                   'tar_stop_prob': stop_prob[:, 1:]}
        return self._train_step_(inp, targets)
    
    def predict(self, inp, max_length=50):
        inp = tf.expand_dims(inp, 0)
        output = tf.expand_dims(self.start_vec, 0)  # shape: (1, 1, mel_channels)
        predictions = {}
        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, output)
            predictions = self.call(
                inp, output, False, enc_padding_mask, combined_mask, dec_padding_mask
            )
            output = tf.concat([output, predictions['final_output'][:1, -1:, :]], axis=-2)
            stop_pred = predictions['stop_prob'][:, -1]
            if int(tf.argmax(stop_pred, axis=-1)) == self.stop_prob_index:
                break
        
        output = {'mel': output[0, 1:, :], 'attention_weights': predictions['attention_weights']}
        return output
    
    def predict_with_target(self, inp, tar, MAX_LENGTH=50):
        import numpy as np
        
        out = {}
        tar_in = {}
        assert np.allclose(inp[0:1, 0, :], self.start_vec), 'Append start vector to input'
        tar_in['own'] = tf.expand_dims(self.start_vec, 0)
        if tar is not None:
            tar_in['train'] = tar[:, :MAX_LENGTH, :]
            out['TE'] = tf.expand_dims(self.start_vec, 0)
        
        for i in range(MAX_LENGTH):
            # if i % 50 == 0:
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_in['own'])
            model_out = self.call(
                inp, tar_in['own'], False, enc_padding_mask, combined_mask, dec_padding_mask
            )  # , apply_conv=False
            # )
            tar_in['own'] = tf.concat([tf.expand_dims(self.start_vec, 0), model_out['final_output']], axis=-2)
            out['own'] = tar_in['own']
            
            if tar is not None:
                tar_in['TE'] = tar[:, 0: i + 1, :]
                model_out = self.call(
                    inp, tar_in['TE'], False, enc_padding_mask, combined_mask, dec_padding_mask
                )  # , apply_conv=False
                # )
                out['TE'] = tf.concat([out['TE'], model_out['final_output'][0:1, -1:, :]], axis=-2)
            
            stop_prob = model_out['stop_prob'][:, -1]
            if int(tf.argmax(stop_prob, axis=-1)) == self.stop_prob_index:
                break
        
        # out['own'] = self.out_module.postnet(out['own'][0:1, 1:, :], training=False)
        out['own'] = out['own'][0:1, 1:, :]
        
        if tar is not None:
            out['TE'] = out['TE'][0:1, 1:, :]
            # out['TE'] = self.out_module.postnet(out['TE'][0:1, 1:, :], training=False)
            
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_in['train'])
            model_out = self.call(
                inp, tar_in['train'], False, enc_padding_mask, combined_mask, dec_padding_mask
            )  # , apply_conv=True
            # )
            out['train'] = model_out['final_output']
        
        return out


class MelTextTransformer(Transformer):
    
    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 decoder_postnet,
                 encoder,
                 decoder,
                 start_token_index,
                 end_token_index,
                 mel_channels):
        super(MelTextTransformer, self).__init__(encoder_prenet, decoder_prenet, encoder, decoder, decoder_postnet)
        self.start_token_index = start_token_index
        self.end_token_index = end_token_index
        self.train_step = tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float64),
                tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            ]
        )(self._train_step)
        self.encoder_seq_dim = -2
        self.decoder_seq_dim = -1
    
    def _eval_loss(self, targets: dict, model_out: dict):
        loss = self.loss(targets['real'], model_out['final_output'])
        return loss
    
    def _train_step(self, inp, tar):
        targets = {'input': tar[:, :-1],
                   'real': tar[:, 1:]}
        return self._train_step_(inp, targets)
    
    def predict(self, encoded_inp_sentence, MAX_LENGTH=100):
        encoder_input = tf.expand_dims(encoded_inp_sentence, 0)
        decoder_input = [self.start_token_index]
        output = tf.expand_dims(decoder_input, 0)
        out_dict = {}
        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(encoder_input, output)
            model_out = self.__call__(
                encoder_input,
                target_vecs=output,
                training=False,
                enc_padding_mask=enc_padding_mask,
                look_ahead_mask=combined_mask,
                dec_padding_mask=dec_padding_mask,
            )
            predictions = model_out['final_output'][:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            output = tf.concat([output, predicted_id], axis=-1)
            out_dict = {'output': tf.squeeze(output, axis=0),
                        'attn_weights': model_out['attention_weights'],
                        'logits': predictions}
            if predicted_id == self.end_token_index:
                break
        return out_dict
