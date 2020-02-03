import tensorflow as tf


def norm_tensor(tensor):
    return tf.math.divide(
        tf.math.subtract(
            tensor,
            tf.math.reduce_min(tensor)
        ),
        tf.math.subtract(
            tf.math.reduce_max(tensor),
            tf.math.reduce_min(tensor)
        )
    )


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
