import os
import tensorflow as tf


def random_mel_mask(tensor, mask_prob):
    tensor_shape = tf.shape(tensor)
    mask_floats = tf.random.uniform((tensor_shape[0], tensor_shape[1]))
    mask = tf.cast(mask_floats > mask_prob, tf.float32)
    mask = tf.expand_dims(mask, -1)
    mask = tf.broadcast_to(mask, tensor_shape)
    masked_tensor = tensor * mask
    return masked_tensor


def random_text_mask(tensor, mask_prob):
    tensor_shape = tf.shape(tensor)
    mask_floats = tf.random.uniform((tensor_shape[0], tensor_shape[1]))
    mask = tf.cast(mask_floats > mask_prob, tf.int64)
    masked_tensor = tensor * mask
    return masked_tensor

# TODO: deprecated, remove soon
def preprocess_mel(mel,
                   start_vec,
                   end_vec,
                   clip_min=1e-5,
                   clip_max=float('inf')):
    norm_mel = tf.cast(mel, tf.float32)
    norm_mel = tf.math.log(tf.clip_by_value(norm_mel, clip_value_min=clip_min, clip_value_max=clip_max))
    norm_mel = tf.concat([start_vec, norm_mel, end_vec], 0)
    return norm_mel


def load_files(metafile,
               meldir,
               num_samples=None):
    samples = []
    count = 0
    alphabet = set()
    with open(metafile, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l_split = l.split('|')
            text = l_split[-1].strip().lower()
            mel_file = os.path.join(str(meldir), l_split[0] + '.npy')
            samples.append((text, mel_file))
            alphabet.update(list(text))
            count += 1
            if num_samples is not None and count > num_samples:
                break
        alphabet = sorted(list(alphabet))
        return samples, alphabet