import tensorflow as tf
from src.layers import Encoder, Decoder
from utils import create_masks, create_mel_masks, masked_loss_function


class TextTransformer(tf.keras.Model):
    """
	(vocab_size - 1): start token
	vocab_size: end token.
	"""

    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, pe_target, vocab_size: dict, rate=0.1):
        super(TextTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.tokens = {}
        assert 'in' in list(vocab_size.keys()), 'Provide "in" input vocab_size'
        assert 'out' in list(vocab_size.keys()), 'Provide "out" output vocab_size'
        for type in ['in', 'out']:
            self.tokens[type] = {'start': vocab_size[type] - 2, 'end': vocab_size[type] - 1}

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, input_vocab_size = vocab_size['in'], rate=rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_target, target_vocab_size = vocab_size['out'], rate=rate)
        self.final_layer = tf.keras.layers.Dense(vocab_size['out'])

    def call(self, inp, target, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(target, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

    def prepare_for_training(self, loss_object):
        self.loss_object = loss_object

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64),]
    )
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = self.__call__(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = masked_loss_function(tar_real, predictions, loss_object=self.loss_object)
        gradients = tape.gradient(loss, self.trainable_variables)

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
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, pe_target, start_vec, stop_vec, rate=0.1):
        super(MelTransformer, self).__init__()
        self.tokens = {'start': start_vec, 'stop': stop_vec}
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate=rate, embed=False)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_target, rate=rate, embed=False)

    def call(self, inp, target, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(target, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

    def prepare_for_training(self, loss_object):
        self.loss_object = loss_object

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64),]
    )
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_mel_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = self.__call__(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = masked_loss_function(tar_real, predictions, loss_object=self.loss_object)
        gradients = tape.gradient(loss, self.trainable_variables)

        return gradients, loss, tar_real, predictions

    def predict(self, encoded_inp_sentence, MAX_LENGTH=40):
        encoder_input = tf.expand_dims(encoded_inp_sentence, 0)
        decoder_input = [self.tokens['start']]
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
            if predicted_id == self.tokens['stop']:
                return tf.squeeze(output, axis=0), attention_weights

            output = tf.concat([output, predicted_id], axis=-1)
        out_dict = {'output': tf.squeeze(output, axis=0), 'attn_weights': attention_weights, 'logits': predictions}
        # return tf.squeeze(output, axis=0), attention_weights, predictions
        return out_dict
