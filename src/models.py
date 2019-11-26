import tensorflow as tf
from src.layers import Encoder, Decoder, SpeechOutModule, PointWiseFFN
from utils import create_masks, create_mel_masks, masked_loss_function, weighted_sum_losses


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

        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            pe_input,
            prenet=tf.keras.layers.Embedding(vocab_size['in'], d_model),
            rate=rate,
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            pe_target,
            prenet=tf.keras.layers.Embedding(vocab_size['out'], d_model),
            rate=rate,
        )
        self.final_layer = tf.keras.layers.Dense(vocab_size['out'])

    def call(self, inp, target, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(
            target, enc_output, training, look_ahead_mask, dec_padding_mask
        )
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

    def prepare_for_training(self, loss_function, optimizer):
        self.loss_function = loss_function
        self.optimizer = optimizer

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ]
    )
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = self.__call__(
                inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask
            )
            loss = masked_loss_function(tar_real, predictions, loss_object=self.loss_function)
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
        out_dict = {
            'output': tf.squeeze(output, axis=0),
            'attn_weights': attention_weights,
            'logits': predictions,
        }
        # return tf.squeeze(output, axis=0), attention_weights, predictions
        return out_dict


class MelTransformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        pe_input,
        pe_target,
        start_vec,
        mel_channels=80,
        conv_filters=256,
        postnet_conv_layers=5,
        postnet_kernel_size=5,
        rate=0.1,
    ):
        super(MelTransformer, self).__init__()
        self.start_vec = start_vec
        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            pe_input,
            prenet=PointWiseFFN(d_model=d_model, dff=dff),
            rate=rate,
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            pe_target,
            prenet=PointWiseFFN(d_model=d_model, dff=dff),
            rate=rate,
        )
        self.out_module = SpeechOutModule(
            mel_channels=mel_channels,
            conv_filters=conv_filters,
            conv_layers=postnet_conv_layers,
            kernel_size=postnet_kernel_size,
        )

    def call(self, inp, target, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(
            target, enc_output, training, look_ahead_mask, dec_padding_mask
        )
        final_output, stop_prob = self.out_module(dec_output)
        return final_output, attention_weights, stop_prob

    def prepare_for_training(self, losses, loss_coeffs):
        self.loss_list = losses
        self.loss_coeffs = loss_coeffs

    # @tf.function(
    # input_signature=[tf.TensorSpec(shape=(None, None, 80), dtype=tf.float64), tf.TensorSpec(shape=(None, None, 80), dtype=tf.float64),]
    # )
    def train_step(self, inp, tar, stop_prob):
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]
        tar_stop_prob = stop_prob[:, 1:, :]
        enc_padding_mask, combined_mask, dec_padding_mask = create_mel_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _, stop_prob = self.__call__(
                inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask
            )
            loss = weighted_sum_losses(
                (tar_real, tar_stop_prob), (predictions, stop_prob), self.loss_list, self.loss_coeffs
            )
            # loss = masked_loss_function(tar_real, predictions, loss_object=self.loss_object)

        gradients = tape.gradient(loss, self.trainable_variables)
        return gradients, loss, tar_real, predictions, stop_prob

    def predict(self, encoded_inp_sentence, MAX_LENGTH=40, stop_tolerance=0.5):
        encoder_input = tf.expand_dims(encoded_inp_sentence, 0)
        decoder_input = [self.start_vec]
        output = tf.cast(tf.expand_dims(decoder_input, 0), tf.float32)

        for i in range(MAX_LENGTH):
            print('i {}'.format(i))
            enc_padding_mask, combined_mask, dec_padding_mask = create_mel_masks(encoder_input, output)
            predictions, attention_weights, stop_probs = self.call(
                encoder_input,
                target=output,
                training=False,
                enc_padding_mask=enc_padding_mask,
                look_ahead_mask=combined_mask,
                dec_padding_mask=dec_padding_mask,
            )
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            output = tf.concat([output, predictions], axis=-2)
            if stop_probs[-1][-1] > stop_tolerance:
                break

        out_dict = {
            'output': tf.squeeze(output, axis=0),
            'attn_weights': attention_weights,
            'logits': predictions,
            'stop_probs': tf.squeeze(stop_probs),
        }

        return out_dict
