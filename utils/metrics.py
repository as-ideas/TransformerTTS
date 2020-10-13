import numpy as np
import tensorflow as tf

from utils.spectrogram_ops import mel_lengths


def attention_score(att, mels, r, mel_padding_value=0.):
    """
    returns a tuple of scores (loc_score, sharp_score), where loc_score measures monotonicity and
    sharp_score measures the sharpness of attention peaks
    attn_weights : [N, n_heads, mel_dim, phoneme_dim]
    """
    
    assert len(tf.shape(mels)) == 3 # [N, mel_max, mel_channels]
    assert len(tf.shape(att)) == 4
    
    mel_len = mel_lengths(mels, mel_padding_value) // r  # [N]
    mask = tf.range(tf.shape(att)[2])[None, :] < mel_len[:, None]
    mask = tf.cast(mask, tf.int32)[:, None, :]  # [N, 1, mel_dim]
    
    # distance between max (jumpiness)
    max_loc = tf.argmax(att, axis=3)  # [N, n_heads, mel_max]
    max_loc_diff = tf.abs(max_loc[:, :, 1:] - max_loc[:, :, :-1])  # [N, h_heads, mel_max - 1]
    loc_score = tf.cast(max_loc_diff >= 0, tf.int32) * tf.cast(max_loc_diff <= r, tf.int32)  # [N, h_heads, mel_max - 1]
    loc_score = tf.reduce_sum(loc_score * mask[:, :, 1:], axis=-1)
    loc_score = loc_score / (mel_len - 1)[:, None]
    
    # variance
    max_loc = tf.reduce_max(att, axis=3)  # [N, n_heads, mel_dim]
    peak_score = tf.reduce_mean(max_loc * tf.cast(mask, tf.float32), axis=-1)
    return loc_score, peak_score


def weight_mask(attention_weights):
    """ exponential loss mask based on distance from approximate diagonal"""
    max_m, max_n = attention_weights.shape
    i = np.tile(np.arange(max_n), (max_m, 1)) / max_n
    j = np.swapaxes(np.tile(np.arange(max_m), (max_n, 1)), 0, 1) / max_m
    return np.sqrt(np.square(i - j))
