import tensorflow as tf


def masked_crossentropy(targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)
    return loss


def masked_mean_squared_error(targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    mse = tf.keras.losses.MeanSquaredError()
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    mask = tf.reduce_max(mask, axis=-1)
    loss = mse(targets, logits, sample_weight=mask)
    return loss


def masked_binary_crossentropy(targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    bc = tf.keras.losses.BinaryCrossentropy(reduction='none')
    mask = tf.math.logical_not(tf.math.equal(targets, -1))
    mask = tf.cast(mask, dtype=tf.int64)
    loss_ = bc(targets, logits)
    loss_ *= mask
    return tf.reduce_mean(loss_)
