import tensorflow as tf

class GST(tf.keras.layers.Layer):
    def __init__(self, model_size: int, num_heads: int, token_num):
        super().__init__()
        self.reference_encoder = ReferenceEncoder(model_size=model_size)
        self.stl = STL(token_num=token_num, embed_dims=model_size, num_heads=num_heads)
        
    def call(self, inputs, **kwargs):
        ref_embedding = self.reference_encoder(inputs)
        style_embedding, attention_weights = self.stl(ref_embedding)
        return style_embedding, attention_weights
        

class ReferenceEncoder(tf.keras.layers.Layer):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''
    
    def __init__(self, model_size):
        super().__init__()
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        self.K = len(ref_enc_filters)
        self.convs = [tf.keras.layers.Conv2D(filters=ref_enc_filters[i],
                                             kernel_size=(3, 3),
                                             strides=(2, 2),
                                             padding='same'
                                             ) for i in range(self.K)]
        self.bns = [tf.keras.layers.BatchNormalization() for _ in range(self.K)]
        self.relus = [tf.keras.layers.ReLU() for _ in range(self.K)]
        self.gru = tf.keras.layers.GRU(units=model_size // 2, dropout=0.1)
        self.out_proj = tf.keras.layers.Dense(model_size)
    
    def call(self, inputs):
        out = tf.expand_dims(inputs, -1)  # [N, mel_len, mel_channels, 1]
        for i in tf.range(self.K):
            out = self.convs[i](out)
            out = self.bns[i](out)
            out = self.relus[i](out)  # [N, mel_len//2^K, mel_channels//2^K, 128]
        batch_size, time_step = tf.shape(out)[0:2]
        out = tf.reshape(out, (batch_size, time_step, -1))  # [N, mel_len//2^K, mel_channels//2^K * 128]
        out = self.gru(out)  # [N, model_size//2]
        out = self.out_proj(out)  # [N, model_size]
        return tf.keras.activations.tanh(out)


class STL(tf.keras.layers.Layer):
    '''
    inputs --- [N, E//2]
    '''
    
    def __init__(self, token_num: int, embed_dims: int, num_heads: int):
        super().__init__()
        self.tokens = tf.Variable(tf.initializers.GlorotUniform()(shape=(token_num, embed_dims // num_heads)))
        self.attention = StyleAttention(model_dim=embed_dims, num_heads=num_heads)
    
    def call(self, inputs):
        N = tf.shape(inputs)[0]
        query = tf.expand_dims(inputs, 1) # (N, 1, embed_dims)
        # TODO: Add cosine similarity loss for tokens instead of tanh
        keys = tf.keras.activations.tanh(self.tokens)
        keys = tf.tile(tf.expand_dims(keys, 0), [N, 1, 1])
        style_embed, attn_score = self.attention(keys, query=query)
        return style_embed, attn_score
    
    # def forward_with_scores(self, attention_scores, batch_size):
    #     keys = tf.keras.activations.tanh(self.embed).unsqueeze(0).expand(batch_size, -1,
    #                                                                      -1)  # [N, token_num, E // num_heads]
    #     style_embed, attn_score = self.attention.forward_with_scores(keys, scores=attention_scores)
    #
    #     return style_embed, attn_score


class StyleAttention(tf.keras.layers.Layer):
    
    def __init__(self, model_dim: int, num_heads: int, **kwargs):
        super(StyleAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.model_dim = model_dim
        
        assert model_dim % self.num_heads == 0
        
        self.depth = model_dim // self.num_heads
        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
    
    def split_heads(self, x, batch_size: int):
        """ Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, keys, query):
        batch_size = tf.shape(query)[0]
        
        q = self.wq(query)  # (batch_size, seq_len, model_dim)
        k = self.wk(keys)  # (batch_size, seq_len, model_dim)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        attention_weights = tf.squeeze(attention_weights)
        attention_values = tf.matmul(attention_weights, keys) # multiply attention scalars with token embeddings directly
        output = tf.reshape(attention_values, (batch_size, -1))
        return output, attention_weights

if __name__ == '__main__':
    bs = 2
    mel_len = 100
    mel_channels = 80
    model_size = 256
    num_heads = 4
    token_num = 10
    mel_batch = tf.random.uniform((bs, mel_len, mel_channels))
    gst = GST(model_size=model_size, num_heads=num_heads, token_num=token_num)
    style_embed, attn_weights = gst(mel_batch)
    assert all(tf.shape(attn_weights) == (bs, num_heads, token_num))
    assert all(tf.shape(style_embed) == (bs, model_size))