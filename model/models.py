import tensorflow as tf
from model.transformer_utils import create_masks, create_mel_masks, weighted_sum_losses, create_mel_text_masks


class TextTransformer(tf.keras.Model):

    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 decoder_postnet,
                 encoder,
                 decoder,
                 start_token_index,
                 end_token_index):
        super(TextTransformer, self).__init__()
        self.start_token_index = start_token_index
        self.end_token_index = end_token_index
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
        enc_output = self.encoder(enc_input, training, enc_padding_mask)
        dec_input = self.decoder_prenet(targets)
        dec_output, attention_weights = self.decoder(inputs=dec_input,
                                                     enc_output=enc_output,
                                                     training=training,
                                                     look_ahead_mask=look_ahead_mask,
                                                     padding_mask=dec_padding_mask)
        final_output = self.decoder_postnet(dec_output)
        return final_output, attention_weights

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                         tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
    )
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = self.__call__(inputs=inp,
                                           targets=tar_inp,
                                           training=True,
                                           enc_padding_mask=enc_padding_mask,
                                           look_ahead_mask=combined_mask,
                                           dec_padding_mask=dec_padding_mask)
            loss = self.loss(tar_real, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return gradients, loss, tar_real, predictions

    def predict(self, encoded_inp_sentence, MAX_LENGTH=40):
        encoder_input = tf.expand_dims(encoded_inp_sentence, 0)
        decoder_input = [self.start_token_index]
        output = tf.expand_dims(decoder_input, 0)
        out_dict = {}
        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
            predictions, attention_weights = self.__call__(
                inputs=encoder_input,
                targets=output,
                training=False,
                enc_padding_mask=enc_padding_mask,
                look_ahead_mask=combined_mask,
                dec_padding_mask=dec_padding_mask,
            )
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            output = tf.concat([output, predicted_id], axis=-1)
            out_dict = {'output': tf.squeeze(output, axis=0), 'attn_weights': attention_weights, 'logits': predictions}
            if predicted_id == self.end_token_index:
                break
        return out_dict


class MelTransformer(tf.keras.Model):

    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 encoder,
                 decoder,
                 decoder_postnet,
                 start_vec,
                 stop_prob_index):
        super(MelTransformer, self).__init__()
        self.start_vec = tf.cast(start_vec, tf.float32)
        self.encoder_prenet = encoder_prenet
        self.decoder_prenet = decoder_prenet
        self.encoder = encoder
        self.decoder = decoder
        self.speech_out_module = decoder_postnet
        self.stop_prob_index = stop_prob_index
        self.train_step = tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, None, decoder_postnet.mel_channels), dtype=tf.float64),
                tf.TensorSpec(shape=(None, None, decoder_postnet.mel_channels), dtype=tf.float64),
                tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            ]
        )(self._train_step)

    def call(self,
             input_vecs,
             target_vecs,
             training,
             enc_padding_mask,
             look_ahead_mask,
             dec_padding_mask):
        enc_input = self.encoder_prenet(input_vecs)
        enc_output = self.encoder(inputs=enc_input,
                                  training=training,
                                  mask=enc_padding_mask)
        dec_input = self.decoder_prenet(target_vecs)
        dec_output, attention_weights = self.decoder(inputs=dec_input,
                                                     enc_output=enc_output,
                                                     training=training,
                                                     look_ahead_mask=look_ahead_mask,
                                                     padding_mask=dec_padding_mask)
        mel_linear, final_output, stop_prob = self.speech_out_module(inputs=dec_output, training=training)
        return {
            'mel_linear': mel_linear,
            'final_output': final_output,
            'stop_prob': stop_prob,
            'attention_weights': attention_weights,
            'decoder_output': dec_output,
        }

    def _train_step(self, inp, tar, stop_prob):
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]
        tar_stop_prob = stop_prob[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_mel_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            model_out = self.__call__(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss, loss_vals = weighted_sum_losses(
                (tar_real, tar_stop_prob, tar_real),
                (model_out['final_output'], model_out['stop_prob'], model_out['mel_linear']),
                self.loss,
                self.loss_weights,
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        out = {
            'mel_linear': model_out['mel_linear'],
            'final_output': model_out['final_output'],
            'stop_prob': model_out['stop_prob'],
            'attention_weights': model_out['attention_weights'],
            'decoder_output': model_out['decoder_output'],
            'loss': loss,
            'loss_vals': loss_vals,
        }
        return out

    def predict(self, inp, max_length=50):
        inp = tf.expand_dims(inp, 0)
        output = tf.expand_dims(self.start_vec, 0)  # shape: (1, 1, mel_channels)
        predictions = {}
        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_mel_masks(inp, output)
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
            enc_padding_mask, combined_mask, dec_padding_mask = create_mel_masks(inp, tar_in['own'])
            model_out = self.call(
                inp, tar_in['own'], False, enc_padding_mask, combined_mask, dec_padding_mask
            )  # , apply_conv=False
            # )
            tar_in['own'] = tf.concat([tf.expand_dims(self.start_vec, 0), model_out['final_output']], axis=-2)
            out['own'] = tar_in['own']

            if tar is not None:
                tar_in['TE'] = tar[:, 0 : i + 1, :]
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

            enc_padding_mask, combined_mask, dec_padding_mask = create_mel_masks(inp, tar_in['train'])
            model_out = self.call(
                inp, tar_in['train'], False, enc_padding_mask, combined_mask, dec_padding_mask
            )  # , apply_conv=True
            # )
            out['train'] = model_out['final_output']

        return out


class MelTextTransformer(tf.keras.Model):

    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 decoder_postnet,
                 encoder,
                 decoder,
                 start_token_index,
                 end_token_index,
                 mel_channels):
        super(MelTextTransformer, self).__init__()
        self.start_token_index = start_token_index
        self.end_token_index = end_token_index
        self.encoder_prenet = encoder_prenet
        self.decoder_prenet = decoder_prenet
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_postnet = decoder_postnet
        #self.train_step = self._train_step
        self.train_step = tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float64),
                tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            ]
        )(self._train_step)

    def call(self,
             inputs,
             targets,
             training,
             enc_padding_mask,
             look_ahead_mask,
             dec_padding_mask):
        enc_input = self.encoder_prenet(inputs)
        enc_output = self.encoder(enc_input, training, enc_padding_mask)
        dec_input = self.decoder_prenet(targets)
        dec_output, attention_weights = self.decoder(inputs=dec_input,
                                                     enc_output=enc_output,
                                                     training=training,
                                                     look_ahead_mask=look_ahead_mask,
                                                     padding_mask=dec_padding_mask)
        final_output = self.decoder_postnet(dec_output)
        return final_output, attention_weights

    def _train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_mel_text_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = self.__call__(inputs=inp,
                                           targets=tar_inp,
                                           training=True,
                                           enc_padding_mask=enc_padding_mask,
                                           look_ahead_mask=combined_mask,
                                           dec_padding_mask=dec_padding_mask)
            loss = self.loss(tar_real, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return gradients, loss, tar_real, predictions

    def predict(self, encoded_inp_sentence, MAX_LENGTH=100):
        encoder_input = tf.expand_dims(encoded_inp_sentence, 0)
        decoder_input = [self.start_token_index]
        output = tf.expand_dims(decoder_input, 0)
        out_dict = {}
        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_mel_text_masks(encoder_input, output)
            predictions, attention_weights = self.__call__(
                inputs=encoder_input,
                targets=output,
                training=False,
                enc_padding_mask=enc_padding_mask,
                look_ahead_mask=combined_mask,
                dec_padding_mask=dec_padding_mask,
            )
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            output = tf.concat([output, predicted_id], axis=-1)
            out_dict = {'output': tf.squeeze(output, axis=0), 'attn_weights': attention_weights, 'logits': predictions}
            if predicted_id == self.end_token_index:
                break
        return out_dict