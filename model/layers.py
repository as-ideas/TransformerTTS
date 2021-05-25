import tensorflow as tf

from model.transformer_utils import positional_encoding


class CNNResNorm(tf.keras.layers.Layer):
    """
    Module used in attention blocks, after MHA
    """
    
    def __init__(self,
                 filters: list,
                 kernel_size: int,
                 inner_activation: str,
                 padding: str,
                 dout_rate: float):
        super(CNNResNorm, self).__init__()
        self.n_layers = len(filters)
        self.convolutions = [tf.keras.layers.Conv1D(filters=f,
                                                    kernel_size=kernel_size,
                                                    padding=padding)
                             for f in filters[:-1]]
        self.inner_activations = [tf.keras.layers.Activation(inner_activation) for _ in range(self.n_layers - 1)]
        self.last_conv = tf.keras.layers.Conv1D(filters=filters[-1],
                                                kernel_size=kernel_size,
                                                padding=padding)
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(rate=dout_rate)
    
    def call_convs(self, x):
        for i in range(0, self.n_layers - 1):
            x = self.convolutions[i](x)
            x = self.inner_activations[i](x)
        return x
    
    def call(self, inputs, training):
        x = self.call_convs(inputs)
        x = self.last_conv(x)
        x = self.dropout(x, training=training)
        return self.normalization(inputs + x)


class TransposedCNNResNorm(tf.keras.layers.Layer):
    """
    Module used in attention blocks, after MHA
    """
    
    def __init__(self,
                 filters: list,
                 kernel_size: int,
                 inner_activation: str,
                 padding: str,
                 dout_rate: float):
        super(TransposedCNNResNorm, self).__init__()
        self.n_layers = len(filters)
        self.convolutions = [tf.keras.layers.Conv1D(filters=f,
                                                    kernel_size=kernel_size,
                                                    padding=padding)
                             for f in filters[:-1]]
        self.inner_activations = [tf.keras.layers.Activation(inner_activation) for _ in range(self.n_layers - 1)]
        self.last_conv = tf.keras.layers.Conv1D(filters=filters[-1],
                                                kernel_size=kernel_size,
                                                padding=padding)
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(rate=dout_rate)
    
    def call_convs(self, x):
        for i in range(0, self.n_layers - 1):
            x = self.convolutions[i](x)
            x = self.inner_activations[i](x)
        return x
    
    def call(self, inputs, training):
        x = tf.transpose(inputs, (0, 1, 2))
        x = self.call_convs(x)
        x = self.last_conv(x)
        x = tf.transpose(x, (0, 1, 2))
        x = self.dropout(x, training=training)
        return self.normalization(inputs + x)


class FFNResNorm(tf.keras.layers.Layer):
    """
    Module used in attention blocks, after MHA
    """
    
    def __init__(self,
                 model_dim: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(FFNResNorm, self).__init__(**kwargs)
        self.d1 = tf.keras.layers.Dense(dense_hidden_units, 'relu')
        self.d2 = tf.keras.layers.Dense(model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.last_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, training):
        ffn_out = self.d1(x)
        ffn_out = self.d2(ffn_out)  # (batch_size, input_seq_len, model_dim)
        ffn_out = self.dropout(ffn_out, training=training)
        return self.last_ln(ffn_out + x)


class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, model_dim: int, num_heads: int, dropout: float, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.model_dim = model_dim
        
        assert model_dim % self.num_heads == 0
        
        self.depth = model_dim // self.num_heads
        
        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dense = tf.keras.layers.Dense(model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout)
    
    def split_heads(self, x, batch_size: int):
        """ Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q_in, mask, training):
        batch_size = tf.shape(q_in)[0]
        
        q = self.wq(q_in)  # (batch_size, seq_len, model_dim)
        k = self.wk(k)  # (batch_size, seq_len, model_dim)
        v = self.wv(v)  # (batch_size, seq_len, model_dim)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        scaled_attention, attention_weights = self.attention([q, k, v, mask], training=training)
        
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.model_dim))  # (batch_size, seq_len_q, model_dim)
        concat_query = tf.concat([q_in, concat_attention], axis=-1)
        output = self.dense(concat_query)  # (batch_size, seq_len_q, model_dim)
        output = self.dropout(output, training=training)
        return output, attention_weights


class ScaledDotProductAttention(tf.keras.layers.Layer):
    """ Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_k, depth)
            v: value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable
                  to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
            output, attention_weights
      """
    
    def __init__(self, dropout: float):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
    
    def call(self, inputs, training=False):
        q, k, v, mask = inputs
        
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += mask * -1e9  # TODO: add mask expansion here and remove from create padding mask
        
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        attention_weights = self.dropout(attention_weights, training=training)
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        
        return output, attention_weights


class SelfAttentionResNorm(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 **kwargs):
        super(SelfAttentionResNorm, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(model_dim, num_heads, dropout=dropout_rate)
        self.last_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, training, mask):
        attn_out, attn_weights = self.mha(x, x, x, mask, training=training)  # (batch_size, input_seq_len, model_dim)
        return self.last_ln(attn_out + x), attn_weights


class SelfAttentionDenseBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(SelfAttentionDenseBlock, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.ffn = FFNResNorm(model_dim, dense_hidden_units, dropout_rate=dropout_rate)
    
    def call(self, x, training, mask):
        attn_out, attn_weights = self.sarn(x, mask=mask, training=training)
        dense_mask = 1. - tf.squeeze(mask, axis=(1, 2))[:, :, None]
        attn_out = attn_out * dense_mask
        return self.ffn(attn_out, training=training) * dense_mask, attn_weights


class SelfAttentionConvBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 conv_filters: list,
                 kernel_size: int,
                 conv_activation: str,
                 transposed_convs: bool,
                 **kwargs):
        super(SelfAttentionConvBlock, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        if transposed_convs:
            self.conv = TransposedCNNResNorm(filters=conv_filters,
                                             kernel_size=kernel_size,
                                             inner_activation=conv_activation,
                                             dout_rate=dropout_rate,
                                             padding='same')
        else:
            self.conv = CNNResNorm(filters=conv_filters,
                                   kernel_size=kernel_size,
                                   inner_activation=conv_activation,
                                   dout_rate=dropout_rate,
                                   padding='same')
    
    def call(self, x, training, mask):
        attn_out, attn_weights = self.sarn(x, mask=mask, training=training)
        conv_mask = 1. - tf.squeeze(mask, axis=(1, 2))[:, :, None]
        attn_out = attn_out * conv_mask
        conv = self.conv(attn_out, training=training)
        return conv * conv_mask, attn_weights


class SelfAttentionBlocks(tf.keras.layers.Layer):
    def __init__(self,
                 model_dim: int,
                 feed_forward_dimension: int,
                 num_heads: list,
                 maximum_position_encoding: int,
                 conv_filters: list,
                 dropout_rate: float,
                 dense_blocks: int,
                 kernel_size: int,
                 conv_activation: str,
                 transposed_convs: bool = None,
                 **kwargs):
        super(SelfAttentionBlocks, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.pos_encoding_scalar = tf.Variable(1.)
        self.pos_encoding = positional_encoding(maximum_position_encoding, model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoder_SADB = [
            SelfAttentionDenseBlock(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                    dense_hidden_units=feed_forward_dimension, name=f'{self.name}_SADB_{i}')
            for i, n_heads in enumerate(num_heads[:dense_blocks])]
        self.encoder_SACB = [
            SelfAttentionConvBlock(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                   name=f'{self.name}_SACB_{i}', kernel_size=kernel_size,
                                   conv_activation=conv_activation, conv_filters=conv_filters,
                                   transposed_convs=transposed_convs)
            for i, n_heads in enumerate(num_heads[dense_blocks:])]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs, training, padding_mask, reduction_factor=1):
        seq_len = tf.shape(inputs)[1]
        x = self.layernorm(inputs)
        x += self.pos_encoding_scalar * self.pos_encoding[:, :seq_len * reduction_factor:reduction_factor, :]
        x = self.dropout(x, training=training)
        attention_weights = {}
        for i, block in enumerate(self.encoder_SADB):
            x, attn_weights = block(x, training=training, mask=padding_mask)
            attention_weights[f'{self.name}_DenseBlock{i + 1}_SelfAttention'] = attn_weights
        for i, block in enumerate(self.encoder_SACB):
            x, attn_weights = block(x, training=training, mask=padding_mask)
            attention_weights[f'{self.name}_ConvBlock{i + 1}_SelfAttention'] = attn_weights
        
        return x, attention_weights


class CrossAttentionResnorm(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 **kwargs):
        super(CrossAttentionResnorm, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(model_dim, num_heads, dropout=dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, q, k, v, training, mask):
        attn_values, attn_weights = self.mha(v, k=k, q_in=q, mask=mask, training=training)
        out = self.layernorm(attn_values + q)
        return out, attn_weights


class CrossAttentionDenseBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(CrossAttentionDenseBlock, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.carn = CrossAttentionResnorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.ffn = FFNResNorm(model_dim, dense_hidden_units, dropout_rate=dropout_rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.sarn(x, mask=look_ahead_mask, training=training)
        
        attn2, attn_weights_block2 = self.carn(attn1, v=enc_output, k=enc_output,
                                               mask=padding_mask, training=training)
        ffn_out = self.ffn(attn2, training=training)
        return ffn_out, attn_weights_block1, attn_weights_block2

# This is never used.
# class CrossAttentionConvBlock(tf.keras.layers.Layer):
#
#     def __init__(self,
#                  model_dim: int,
#                  num_heads: int,
#                  conv_filters: list,
#                  dropout_rate: float,
#                  kernel_size: int,
#                  conv_padding: str,
#                  conv_activation: str,
#                  **kwargs):
#         super(CrossAttentionConvBlock, self).__init__(**kwargs)
#         self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
#         self.carn = CrossAttentionResnorm(model_dim, num_heads, dropout_rate=dropout_rate)
#         self.conv = CNNResNorm(filters=conv_filters,
#                                kernel_size=kernel_size,
#                                inner_activation=conv_activation,
#                                padding=conv_padding,
#                                dout_rate=dropout_rate)
#
#     def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
#         attn1, attn_weights_block1 = self.sarn(x, mask=look_ahead_mask, training=training)
#
#         attn2, attn_weights_block2 = self.carn(attn1, v=enc_output, k=enc_output,
#                                                mask=padding_mask, training=training)
#         ffn_out = self.conv(attn2, training=training)
#         return ffn_out, attn_weights_block1, attn_weights_block2


class CrossAttentionBlocks(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 feed_forward_dimension: int,
                 num_heads: list,
                 maximum_position_encoding: int,
                 dropout_rate: float,
                 **kwargs):
        super(CrossAttentionBlocks, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.pos_encoding_scalar = tf.Variable(1.)
        self.pos_encoding = positional_encoding(maximum_position_encoding, model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.CADB = [
            CrossAttentionDenseBlock(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                     dense_hidden_units=feed_forward_dimension, name=f'{self.name}_CADB_{i}')
            for i, n_heads in enumerate(num_heads[:-1])]
        self.last_CADB = CrossAttentionDenseBlock(model_dim=model_dim, dropout_rate=dropout_rate,
                                                  num_heads=num_heads[-1],
                                                  dense_hidden_units=feed_forward_dimension,
                                                  name=f'{self.name}_CADB_last')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, enc_output, training, decoder_padding_mask, encoder_padding_mask,
             reduction_factor=1):
        seq_len = tf.shape(inputs)[1]
        x = self.layernorm(inputs)
        x += self.pos_encoding_scalar * self.pos_encoding[:, :seq_len * reduction_factor:reduction_factor, :]
        x = self.dropout(x, training=training)
        attention_weights = {}
        for i, block in enumerate(self.CADB):
            x, _, attn_weights = block(x, enc_output, training, decoder_padding_mask, encoder_padding_mask)
            attention_weights[f'{self.name}_DenseBlock{i + 1}_CrossAttention'] = attn_weights
        x, _, attn_weights = self.last_CADB(x, enc_output, training, decoder_padding_mask, encoder_padding_mask)
        attention_weights[f'{self.name}_LastBlock_CrossAttention'] = attn_weights
        return x, attention_weights


class DecoderPrenet(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(DecoderPrenet, self).__init__(**kwargs)
        self.d1 = tf.keras.layers.Dense(dense_hidden_units,
                                        activation='relu')  # (batch_size, seq_len, dense_hidden_units)
        self.d2 = tf.keras.layers.Dense(model_dim, activation='relu')  # (batch_size, seq_len, model_dim)
        self.rate = tf.Variable(dropout_rate, trainable=False)
        self.dropout_1 = tf.keras.layers.Dropout(self.rate)
        self.dropout_2 = tf.keras.layers.Dropout(self.rate)
    
    def call(self, x, training):
        self.dropout_1.rate = self.rate
        self.dropout_2.rate = self.rate
        x = self.d1(x)
        # use dropout also in inference for positional encoding relevance
        x = self.dropout_1(x, training=training)
        x = self.d2(x)
        x = self.dropout_2(x, training=training)
        return x


class Postnet(tf.keras.layers.Layer):
    
    def __init__(self, mel_channels: int, **kwargs):
        super(Postnet, self).__init__(**kwargs)
        self.mel_channels = mel_channels
        self.stop_linear = tf.keras.layers.Dense(3)
        self.mel_out = tf.keras.layers.Dense(mel_channels)
    
    def call(self, x):
        stop = self.stop_linear(x)
        mel = self.mel_out(x)
        return {
            'mel': mel,
            'stop_prob': stop,
        }


class StatPredictor(tf.keras.layers.Layer):
    def __init__(self,
                 conv_filters: list,
                 kernel_size: int,
                 conv_padding: str,
                 conv_activation: str,
                 dense_activation: str,
                 dropout_rate: float,
                 **kwargs):
        super(StatPredictor, self).__init__(**kwargs)
        self.conv_blocks = CNNDropout(filters=conv_filters,
                                      kernel_size=kernel_size,
                                      padding=conv_padding,
                                      inner_activation=conv_activation,
                                      last_activation=conv_activation,
                                      dout_rate=dropout_rate)
        self.linear = tf.keras.layers.Dense(1, activation=dense_activation)
    
    def call(self, x, training, mask):
        x = x * mask
        x = self.conv_blocks(x, training=training)
        x = self.linear(x)
        return x * mask


class CNNDropout(tf.keras.layers.Layer):
    def __init__(self,
                 filters: list,
                 kernel_size: int,
                 inner_activation: str,
                 last_activation: str,
                 padding: str,
                 dout_rate: float):
        super(CNNDropout, self).__init__()
        self.n_layers = len(filters)
        self.convolutions = [tf.keras.layers.Conv1D(filters=f,
                                                    kernel_size=kernel_size,
                                                    padding=padding)
                             for f in filters[:-1]]
        self.inner_activations = [tf.keras.layers.Activation(inner_activation) for _ in range(self.n_layers - 1)]
        self.last_conv = tf.keras.layers.Conv1D(filters=filters[-1],
                                                kernel_size=kernel_size,
                                                padding=padding)
        self.last_activation = tf.keras.layers.Activation(last_activation)
        self.dropouts = [tf.keras.layers.Dropout(rate=dout_rate) for _ in range(self.n_layers)]
        self.normalization = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(self.n_layers)]
    
    def call_convs(self, x, training):
        for i in range(0, self.n_layers - 1):
            x = self.convolutions[i](x)
            x = self.inner_activations[i](x)
            x = self.normalization[i](x)
            x = self.dropouts[i](x, training=training)
        return x
    
    def call(self, inputs, training):
        x = self.call_convs(inputs, training=training)
        x = self.last_conv(x)
        x = self.last_activation(x)
        x = self.normalization[-1](x)
        x = self.dropouts[-1](x, training=training)
        return x


class Expand(tf.keras.layers.Layer):
    """ Expands a 3D tensor on its second axis given a list of dimensions.
        Tensor should be:
            batch_size, seq_len, dimension
        
        E.g:
        input = tf.Tensor([[[0.54710746 0.8943467 ]
                          [0.7140938  0.97968304]
                          [0.5347662  0.15213418]]], shape=(1, 3, 2), dtype=float32)
        dimensions = tf.Tensor([1 3 2], shape=(3,), dtype=int32)
        output = tf.Tensor([[[0.54710746 0.8943467 ]
                           [0.7140938  0.97968304]
                           [0.7140938  0.97968304]
                           [0.7140938  0.97968304]
                           [0.5347662  0.15213418]
                           [0.5347662  0.15213418]]], shape=(1, 6, 2), dtype=float32)
    """
    
    def __init__(self, model_dim, **kwargs):
        super(Expand, self).__init__(**kwargs)
        self.model_dimension = model_dim
    
    def call(self, x, dimensions):
        dimensions = tf.squeeze(dimensions, axis=-1)
        dimensions = tf.cast(tf.math.round(dimensions), tf.int32)
        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        # build masks from dimensions
        max_dim = tf.math.reduce_max(dimensions)
        tot_dim = tf.math.reduce_sum(dimensions)
        index_masks = tf.RaggedTensor.from_row_lengths(tf.ones(tot_dim), tf.reshape(dimensions, [-1])).to_tensor()
        index_masks = tf.cast(tf.reshape(index_masks, (batch_size, seq_len * max_dim)), tf.float32)
        non_zeros = seq_len * max_dim - tf.reduce_sum(max_dim - dimensions, axis=1)
        # stack and mask
        tiled = tf.tile(x, [1, 1, max_dim])
        reshaped = tf.reshape(tiled, (batch_size, seq_len * max_dim, self.model_dimension))
        mask_reshape = tf.multiply(reshaped, index_masks[:, :, tf.newaxis])
        ragged = tf.RaggedTensor.from_row_lengths(mask_reshape[index_masks > 0], non_zeros)
        return ragged.to_tensor()
