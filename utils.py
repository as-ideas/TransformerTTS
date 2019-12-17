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


def display_attention(attention, layer_name, file=None):
    attention = tf.squeeze(attention[layer_name])
    fig = plt.figure(figsize=(8 * attention.shape[0], 8))
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(1, attention.shape[0], head + 1)
        ax.matshow(attention[head][:-1, :], cmap='viridis')
        ax.set_xlabel('head {}'.format(head + 1))
    plt.tight_layout()
    if not file:
        plt.show()
    else:
        plt.savefig(file)
    plt.close('all')
