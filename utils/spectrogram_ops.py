import tensorflow as tf


def mel_padding_mask(mel_batch, padding_value=0):
    return 1.0 - tf.cast(mel_batch == padding_value, tf.float32)


def mel_lengths(mel_batch, padding_value=0):
    mask = mel_padding_mask(mel_batch, padding_value=padding_value)
    mel_channels = tf.shape(mel_batch)[-1]
    sum_tot = tf.cast(mel_channels, tf.float32) * padding_value
    idxs = tf.cast(tf.reduce_sum(mask, axis=-1) != sum_tot, tf.int32)
    return tf.reduce_sum(idxs, axis=-1)


def phoneme_lengths(phonemes, phoneme_padding=0):
    return tf.reduce_sum(tf.cast(phonemes != phoneme_padding, tf.int32), axis=-1)
