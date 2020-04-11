import tensorflow as tf

from model.transformer_utils import positional_encoding, scaled_dot_product_attention


class PointWiseFFN(tf.keras.layers.Layer):
    
    def __init__(self, model_dim, dense_hidden_units):
        super(PointWiseFFN, self).__init__()
        self.d1 = tf.keras.layers.Dense(dense_hidden_units,
                                        activation='relu')  # (batch_size, seq_len, dense_hidden_units)
        self.d2 = tf.keras.layers.Dense(model_dim)  # (batch_size, seq_len, model_dim)
    
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return x


class DecoderPrenet(tf.keras.layers.Layer):
    
    def __init__(self, model_dim, dense_hidden_units, dropout_rate=0.5):
        super(DecoderPrenet, self).__init__()
        self.d1 = tf.keras.layers.Dense(dense_hidden_units,
                                        activation='relu')  # (batch_size, seq_len, dense_hidden_units)
        self.d2 = tf.keras.layers.Dense(model_dim, activation='relu')  # (batch_size, seq_len, model_dim)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, dropout_rate=0.5):
        self.dropout_1.dropout_rate = dropout_rate
        self.dropout_2.dropout_rate = dropout_rate
        x = self.d1(x)
        # use dropout also in inference for additional noise as suggested in the original tacotron2 paper
        x = self.dropout_1(x, training=True)
        x = self.d2(x)
        x = self.dropout_2(x, training=True)
        return x


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        
        assert model_dim % self.num_heads == 0
        
        self.depth = model_dim // self.num_heads
        
        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)
        
        self.dense = tf.keras.layers.Dense(model_dim)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q_in, mask):
        batch_size = tf.shape(q_in)[0]
        
        q = self.wq(q_in)  # (batch_size, seq_len, model_dim)
        k = self.wk(k)  # (batch_size, seq_len, model_dim)
        v = self.wv(v)  # (batch_size, seq_len, model_dim)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.model_dim))  # (batch_size, seq_len_q, model_dim)
        concat_query = tf.concat([q_in, concat_attention], axis=-1)
        output = self.dense(concat_query)  # (batch_size, seq_len_q, model_dim)
        
        return output, attention_weights


class SelfAttentionResNorm(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, dropout_rate):
        super(SelfAttentionResNorm, self).__init__()
        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask):
        attn_out, attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, model_dim)
        attn_out = self.dropout(attn_out, training=training)
        out = self.ln(x + attn_out)  # (batch_size, input_seq_len, model_dim)
        return out, attn_weights


class FFNResNorm(tf.keras.layers.Layer):
    def __init__(self, model_dim, dense_hidden_units, dropout_rate=0.1):
        super(FFNResNorm, self).__init__()
        self.ffn = PointWiseFFN(model_dim, dense_hidden_units)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, training):
        ffn_out = self.ffn(x)  # (batch_size, input_seq_len, model_dim)
        ffn_out = self.dropout(ffn_out, training=training)
        out = self.ln(x + ffn_out)  # (batch_size, input_seq_len, model_dim)
        
        return out


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, dense_hidden_units, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.ffn = FFNResNorm(model_dim, dense_hidden_units)
    
    def call(self, x, training, mask):
        attn_out, _ = self.sarn(x, mask=mask, training=training)
        return self.ffn(attn_out, training=training)


class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, num_layers, model_dim, num_heads, dense_hidden_units, maximum_position_encoding,
                 dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, model_dim)
        self.enc_layers = [EncoderLayer(model_dim, num_heads, dense_hidden_units, dropout_rate) for _ in
                           range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs, training, mask):
        seq_len = tf.shape(inputs)[1]
        x = inputs * tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, dense_hidden_units, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(model_dim, num_heads)
        self.mha2 = MultiHeadAttention(model_dim, num_heads)
        
        self.ffn = PointWiseFFN(model_dim, dense_hidden_units)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
    
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
    
    def __init__(self, num_layers, model_dim, num_heads, dense_hidden_units, maximum_position_encoding,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, model_dim)
        self.dec_layers = [DecoderLayer(model_dim, num_heads, dense_hidden_units, dropout_rate) for _ in
                           range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(inputs)[1]
        attention_weights = {}
        x = inputs * tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2
        
        return x, attention_weights


class Postnet(tf.keras.layers.Layer):
    
    def __init__(self, mel_channels, conv_filters=256, conv_layers=5, kernel_size=5):
        super(Postnet, self).__init__()
        self.mel_channels = mel_channels
        self.stop_linear = tf.keras.layers.Dense(3)
        self.postnet_conv_layers = PostnetConvLayers(
            out_size=mel_channels, n_filters=conv_filters, n_layers=conv_layers, kernel_size=kernel_size
        )
        self.add_layer = tf.keras.layers.Add()
    
    def call(self, x, training):
        stop = self.stop_linear(x)
        conv_out = self.conv_net(x, training)
        return {
            'mel_linear': x,
            'final_output': conv_out,
            'stop_prob': stop,
        }
    
    def conv_net(self, x, training):
        conv_out = self.postnet_conv_layers(x, training)
        x = self.add_layer([conv_out, x])
        return x


class PostnetConvLayers(tf.keras.layers.Layer):
    
    def __init__(self, out_size, n_filters=256, n_layers=5, kernel_size=5, dropout_prob=0.5):
        super(PostnetConvLayers, self).__init__()
        self.convolutions = [tf.keras.layers.Conv1D(filters=n_filters,
                                                    kernel_size=kernel_size,
                                                    padding='causal',
                                                    activation='tanh')
                             for _ in range(n_layers - 1)]
        self.dropouts = [tf.keras.layers.Dropout(dropout_prob) for _ in range(n_layers - 1)]
        self.last_conv = tf.keras.layers.Conv1D(filters=out_size,
                                                kernel_size=kernel_size,
                                                padding='causal',
                                                activation='linear')
        self.batch_norms = [tf.keras.layers.BatchNormalization() for _ in range(n_layers)]
    
    def call(self, x, training):
        for i in range(0, len(self.convolutions)):
            x = self.convolutions[i](x)
            x = self.batch_norms[i](x, training=training)
            x = self.dropouts[i](x, training=training)
        x = self.last_conv(x)
        x = self.batch_norms[-1](x, training=training)
        return x
