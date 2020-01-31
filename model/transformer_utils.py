import numpy as np
import tensorflow as tf


class CharTokenizer:
    
    def __init__(self, alphabet, start_token='<', end_token='>'):
        self.alphabet = alphabet
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet, start=1)}
        self.idx_to_token[0] = '/'
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.start_token_index = len(self.alphabet) + 1
        self.end_token_index = len(self.alphabet) + 2
        self.vocab_size = len(self.alphabet) + 3
        self.idx_to_token[self.start_token_index] = start_token
        self.idx_to_token[self.end_token_index] = end_token
    
    def encode(self, sentence):
        return [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]

    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence if int(t) in self.idx_to_token])


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
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
    
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9
    
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


def weighted_sum_losses(targets, pred, loss_functions, coeffs):
    total_loss = 0
    loss_vals = []
    for i in range(len(loss_functions)):
        loss = loss_functions[i](targets[i], pred[i])
        loss_vals.append(loss)
        total_loss += coeffs[i] * loss
    return total_loss, loss_vals


# def weighted_sum_losses(tar_real, loss_functions, predictions, coeffs=[1.], tar_stop_prob=None, mel_linear_target=None):
#     total_loss = 0
#     targets = [x for x in [tar_real, tar_stop_prob, mel_linear_target] if x is not None]
#     loss_pairs = zip(targets, predictions)
#     for i, (tar, pred) in enumerate(loss_pairs):
#         loss = loss_functions[i](tar, pred)
#         total_loss += coeffs[i] * loss
#     return loss

def create_text_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, y, x)


def create_mel_padding_mask(seq):
    seq = tf.reduce_sum(seq, axis=-1)
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, y, x)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask