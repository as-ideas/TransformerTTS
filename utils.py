# for display mel
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io


def buffer_mel(ms, sr):
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close('all')
    return buf


def plot_mel(ms, sr, file=None):
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    if not file:
        plt.show()
    else:
        plt.savefig(file)
    plt.close('all')


def plot_attention(outputs, step, info_string=''):
    for k in outputs['attention_weights'].keys():
        for i in range(len(outputs['attention_weights'][k][0])):
            image_batch = norm_tensor(tf.expand_dims(outputs['attention_weights'][k][:, i, :, :], -1))
            tf.summary.image(info_string + k + f' head{i}', image_batch, step=step)


def display_mel(pred, step, info_string='', sr=22050):
    img = tf.transpose(tf.exp(pred))
    buf = buffer_mel(img, sr=sr)
    img_tf = tf.image.decode_png(buf.getvalue(), channels=3)
    img_tf = tf.expand_dims(img_tf, 0)
    tf.summary.image(info_string, img_tf, step=step)


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


