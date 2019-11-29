import librosa
import numpy as np
import time
import tensorflow as tf
import sys
from pathlib import Path

SCRIPT_DIR = Path('.').absolute().parent
# SCRIPT_DIR = Path(__file__).absolute().parent
sys.path.append(SCRIPT_DIR.parent.as_posix())
from src.models import MelTransformer
from utils import display_mel

tf.random.set_seed(10)
np.random.seed(42)

# In[]: VARIABLES
USE_CONV = False
WAV_FILE = '/Users/fcardina/Downloads/LJ001-0173.wav'
WEIGHTS_ID = 'melT_TAILconv'
MEL_CHANNELS = 128
DROPOUT = 0.0
start_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) - 2.0
end_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) + 2.0

params = {
    'num_layers': 1,
    'd_model': 128,
    'num_heads': 1,
    'dff': 128,
    'pe_input': 800,
    'pe_target': 800,
    'start_vec': start_vec,
    'mel_channels': MEL_CHANNELS,
    'conv_filters': 128,
    'postnet_conv_layers': 5,
    'postnet_kernel_size': 5,
    'rate': DROPOUT,
}

# In[]: GET MEL SPECTROGRAM FROM WAV
power_exp = 1
n_fft = 1024
win_length = 1024
MEL_CHANNELS = 128
y, sr = librosa.load('/Users/fcardina/Downloads/LJ001-0173.wav')
ms = librosa.feature.melspectrogram(
    y=y, sr=sr, n_mels=MEL_CHANNELS, power=power_exp, n_fft=n_fft, win_length=win_length, hop_length=256, fmin=0, fmax=8000
)
norm_ms = np.log(ms.clip(1e-5)).T

# In[]: CREATE THE MODEL
melT = MelTransformer(**params)

# In[]:
from utils import create_mel_masks


def custom_prediction(model, inp, tar, start_vec, end_vec, MAX_LENGTH):
    types = ['own', 'TE']
    tar_in = {}
    out = {}
    assert np.allclose(inp[0:1, 0, :], start_vec), 'Append start vector to input'
    assert np.allclose(inp[0:1, -1, :], end_vec), 'Append end vector to input'
    tar_in['own'] = start_vec.copy()
    tar_in['train'] = tar[:, :-1, :]
    out['TE'] = start_vec.copy()
    for i in range(MAX_LENGTH):
        if i % 50 == 0:
            print(i)
        enc_padding_mask, combined_mask, dec_padding_mask = create_mel_masks(inp, tar_in['own'])
        predictions, _, _ = model.call(inp, tar_in['own'], False, enc_padding_mask, combined_mask, dec_padding_mask)
        tar_in['own'] = tf.concat([start_vec.copy(), predictions], axis=-2)
        out['own'] = tar_in['own']
        tar_in['TE'] = tar[:, 0 : i + 1, :]
        # enc_padding_mask, combined_mask, dec_padding_mask = create_mel_masks(inp, tar_in['TE'])
        predictions, _, _ = model.call(inp, tar_in['TE'], False, enc_padding_mask, combined_mask, dec_padding_mask)
        out['TE'] = tf.concat([out['TE'], predictions[0:1, -1:, :]], axis=-2)

    out['own'] = model.out_module.tail(out['own'])
    out['TE'] = model.out_module.tail(out['TE'])

    enc_padding_mask, combined_mask, dec_padding_mask = create_mel_masks(inp, tar_in['train'])
    predictions, _, _ = model.call(inp, tar_in['train'], True, enc_padding_mask, combined_mask, dec_padding_mask)
    out['train'] = tf.concat([start_vec.copy(), predictions], axis=-2)
    return out


# In[]:
out = {}
pred_mel = norm_ms[0:100, :]
pred_mel = tf.concat([start_vec, pred_mel, end_vec], axis=-2)
pred_mel = tf.expand_dims(pred_mel, 0)
display_mel(np.exp(pred_mel[0]).T, sr)

# In[]:
init_pred_epoch = 100
last_pred_epoch = 1600
# In[]:
_ = melT.predict(pred_mel, MAX_LENGTH=1)
for epoch_n in range(init_pred_epoch, last_pred_epoch + 1, 100):
    print(epoch_n)
    melT.load_weights(f'weights/{WEIGHTS_ID}_weights_{epoch_n}.hdf5')
    out[epoch_n] = custom_prediction(
        melT, pred_mel, pred_mel, tf.expand_dims(start_vec, 0).numpy(), tf.expand_dims(end_vec, 0).numpy(), MAX_LENGTH=100
    )

# In[]
init_pred_epoch = 100
last_pred_epoch = 200
# In[]
for epoch_n in range(init_pred_epoch, last_pred_epoch + 1, 100):
    print(epoch_n)
    display_mel(np.exp(out[epoch_n]['own'].numpy()[0].T), sr)
# In[]:
for epoch_n in range(init_pred_epoch, last_pred_epoch + 1, 100):
    print(epoch_n)
    display_mel(np.exp(out[epoch_n]['train'].numpy()[0].T), sr)
# In[]:
for epoch_n in range(init_pred_epoch, last_pred_epoch + 1, 100):
    print(epoch_n)
    display_mel(np.exp(out[epoch_n]['TE'].numpy()[0].T), sr)
# In[]:

A = pred_mel[:, :-1, :]
B = pred_mel
B.shape
A.shape
enc_padding_mask, combined_mask, dec_padding_mask = create_mel_masks(B, A)
out1, _, _ = melT.call(B, A, True, enc_padding_mask, combined_mask, dec_padding_mask)
out2, _, _ = melT.call(B, A, False, enc_padding_mask, combined_mask, dec_padding_mask)

out1


melT.out_module.tail(out2)
