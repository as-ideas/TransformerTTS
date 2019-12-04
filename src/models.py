import tensorflow as tf
from src.layers import Encoder, Decoder, SpeechOutModule, PointWiseFFN
from utils import create_masks, create_mel_masks, masked_loss_function, weighted_sum_losses


class TextTransformer(tf.keras.Model):

    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 encoder,
                 decoder,
                 vocab_size: dict):
        super(TextTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.tokens = {}
        assert 'in' in list(vocab_size.keys()), 'Provide "in" input vocab_size'
        assert 'out' in list(vocab_size.keys()), 'Provide "out" output vocab_size'
        for type in ['in', 'out']:
            self.tokens[type] = {'start': vocab_size[type] - 2, 'end': vocab_size[type] - 1}

        self.encoder_prenet = encoder_prenet
        self.decoder_prenet = decoder_prenet
        self.encoder = encoder
        self.decoder = decoder
        self.final_layer = tf.keras.layers.Dense(vocab_size['out'])

    def call(self, inp, target, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_input = self.encoder_prenet(inp)
        enc_output = self.encoder(enc_input, training, enc_padding_mask)
        dec_input = self.decoder_prenet(target)
        dec_output, attention_weights = self.decoder(dec_input, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
    )
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = self.__call__(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = masked_loss_function(tar_real, predictions, loss_object=self.loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return gradients, loss, tar_real, predictions

    def predict(self, encoded_inp_sentence, MAX_LENGTH=40):
        encoder_input = tf.expand_dims(encoded_inp_sentence, 0)
        decoder_input = [self.tokens['out']['start']]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
            predictions, attention_weights = self.__call__(
                encoder_input,
                target=output,
                training=False,
                enc_padding_mask=enc_padding_mask,
                look_ahead_mask=combined_mask,
                dec_padding_mask=dec_padding_mask,
            )
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            if predicted_id == self.tokens['out']['end']:
                return tf.squeeze(output, axis=0), attention_weights

            output = tf.concat([output, predicted_id], axis=-1)
        out_dict = {'output': tf.squeeze(output, axis=0), 'attn_weights': attention_weights, 'logits': predictions}
        # return tf.squeeze(output, axis=0), attention_weights, predictions
        return out_dict


class MelTransformer(tf.keras.Model):

    def __init__(self,
                 encoder_prenet,
                 decoder_prenet,
                 encoder,
                 decoder,
                 decoder_postnet,
                 start_vec):
        super(MelTransformer, self).__init__()
        self.start_vec = tf.cast(start_vec, tf.float32)
        self.encoder_prenet = encoder_prenet
        self.decoder_prenet = decoder_prenet
        self.encoder = encoder
        self.decoder = decoder
        self.speech_out_module = decoder_postnet
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
        output = {'mel': output[0, 1:, :], 'attention_weights': predictions['attention_weights']}
        return output
