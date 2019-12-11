import tensorflow as tf

from model.transformer_utils import positional_encoding, scaled_dot_product_attention


class PointWiseFFN(tf.keras.layers.Layer):
    
    def __init__(self, d_model, dff):
        super(PointWiseFFN, self).__init__()
        self.d1 = tf.keras.layers.Dense(dff, activation='relu')  # (batch_size, seq_len, dff)
        self.d2 = tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return x


class ReluFeedForward(tf.keras.layers.Layer):
    
    def __init__(self, d_model, dff, dropout_rate=0.5):
        super(ReluFeedForward, self).__init__()
        self.d1 = tf.keras.layers.Dense(dff, activation='relu')  # (batch_size, seq_len, dff)
        self.d2 = tf.keras.layers.Dense(d_model, activation='relu')  # (batch_size, seq_len, d_model)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training=True):
        x = self.d1(x)
        x = self.dropout_1(x, training=training)
        x = self.d2(x)
        x = self.dropout_2(x, training=training)
        return x


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFFN(d_model, dff)
        
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        attn_out, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.ln1(x + attn_out)  # (batch_size, input_seq_len, d_model)
        
        ffn_out = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.ln2(out1 + ffn_out)  # (batch_size, input_seq_len, d_model)
        
        return out2


class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, inputs, training, mask):
        seq_len = tf.shape(inputs)[1]
        x = inputs * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = PointWiseFFN(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.layernorm3(ffn_out + out2)
        
        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, inputs, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(inputs)[1]
        attention_weights = {}
        x = inputs * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2
        
        return x, attention_weights


class TextPostnet(tf.keras.layers.Layer):
    def __init__(self, target_vocab_size):
        super(TextPostnet, self).__init__()
        self.dense_out = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, x):
        return {'final_output': self.dense_out(x)}


class SpeechPostnet(tf.keras.layers.Layer):
    
    def __init__(self, mel_channels, conv_filters=256, conv_layers=5, kernel_size=5):
        super(SpeechPostnet, self).__init__()
        self.mel_channels = mel_channels
        self.mel_linear = tf.keras.layers.Dense(mel_channels)
        self.stop_linear = tf.keras.layers.Dense(3)
        self.speech_conv_layers = SpeechConvLayers(
            out_size=mel_channels, n_filters=conv_filters, n_layers=conv_layers, kernel_size=kernel_size
        )
        self.add_layer = tf.keras.layers.Add()
    
    def call(self, x, training):
        stop = self.stop_linear(x)
        mel_linear = self.mel_linear(x)
        conv_out = self.postnet(mel_linear, training)
        return {
            'mel_linear': mel_linear,
            'final_output': conv_out,
            'stop_prob': stop,
        }
    
    def postnet(self, x, training):
        conv_out = self.speech_conv_layers(x, training)
        x = self.add_layer([conv_out, x])
        return x


class SpeechConvLayers(tf.keras.layers.Layer):
    
    def __init__(self, out_size, n_filters=256, n_layers=5, kernel_size=5):
        super(SpeechConvLayers, self).__init__()
        self.convolutions = [
            tf.keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, padding='causal', activation='relu')
            for _ in range(n_layers - 1)
        ]
        self.last_conv = tf.keras.layers.Conv1D(filters=out_size, kernel_size=kernel_size, padding='causal',
                                                activation='relu')
        self.batch_norms = [tf.keras.layers.BatchNormalization() for _ in range(n_layers)]
    
    def call(self, x, training):
        for i in range(0, len(self.convolutions)):
            x = self.convolutions[i](x)
            x = self.batch_norms[i](x, training=training)
        x = self.last_conv(x)
        x = self.batch_norms[-1](x, training=training)
        return x
