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
            num_layers, d_model, num_heads, dff, pe_input, prenet=tf.keras.layers.Embedding(vocab_size['in'], d_model), rate=rate
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
        dec_output, attention_weights = self.decoder(target, enc_output, training, look_ahead_mask, dec_padding_mask)
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
        self.start_vec = tf.cast(start_vec, tf.float32)
        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, pe_input, prenet=PointWiseFFN(d_model=d_model, dff=dff), rate=rate
        )
        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, pe_target, prenet=PointWiseFFN(d_model=d_model, dff=dff), rate=rate
        )
        self.out_module = SpeechOutModule(
            mel_channels=mel_channels, conv_filters=conv_filters, conv_layers=postnet_conv_layers, kernel_size=postnet_kernel_size
        )

    def call(self, inp, target, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):  # , apply_conv=False):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(target, enc_output, training, look_ahead_mask, dec_padding_mask)
        mel_linear, final_output, stop_prob = self.out_module(dec_output, training=training)  # , apply_conv=apply_conv)
        return {
            'mel_linear': mel_linear,
            'final_output': final_output,
            'stop_prob': stop_prob,
            'attention_weights': attention_weights,
            'dec_output': dec_output,
        }

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, None, 80), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None, 80), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ]
    )
    def train_step(self, inp, tar, stop_prob):
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
            'dec_output': model_out['dec_output'],
            'loss': loss,
            'loss_vals': loss_vals,
        }
        return out

    def predict(self, inp, MAX_LENGTH=50):
        inp = tf.expand_dims(inp, 0)
        output = tf.expand_dims(self.start_vec, 0)  # shape: (1, 1, mel_channels)
        predictions = {}
        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_mel_masks(inp, output)
            predictions = self.call(
                inp, output, False, enc_padding_mask, combined_mask, dec_padding_mask
            )
            output = tf.concat([output, predictions['final_output'][0:1, -1:, :]], axis=-2)
        output = {'mel': output[0, 1:, :], 'attn': predictions['attention_weights']}
        return output

    # TODO: remove
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
