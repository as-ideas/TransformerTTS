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
WAV_FILE = '/Users/fcardina/forge/text-to-speech/data/wav_files/LJ001-0173.wav'
WEIGHTS_ID = 'melT_TAILconv'
MEL_CHANNELS = 128
DROPOUT = 0.0
start_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) - 2.0
end_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) + 2.0


params = {
    'num_layers': 1,
    'd_model': 256,
    'num_heads': 1,
    'dff': 256,
    'pe_input': 800,
    'pe_target': 800,
    'start_vec': start_vec,
    'mel_channels': MEL_CHANNELS,
    'conv_filters': 128,
    'postnet_conv_layers': 3,
    'postnet_kernel_size': 5,
    'rate': DROPOUT,
}
# In[]: GET MEL SPECTROGRAM FROM WAV
power_exp = 1
n_fft = 1024
win_length = 1024
MEL_CHANNELS = 128
y, sr = librosa.load(WAV_FILE)
ms = librosa.feature.melspectrogram(
    y=y, sr=sr, n_mels=MEL_CHANNELS, power=power_exp, n_fft=n_fft, win_length=win_length, hop_length=256, fmin=0, fmax=8000
)
norm_ms = np.log(ms.clip(1e-5)).T

# In[]: CREATE THE MODEL
melT = MelTransformer(**params)

# In[]:
out = {}
pred_mel = norm_ms[0:100, :]
pred_mel = tf.concat([start_vec, pred_mel, end_vec], axis=-2)
pred_mel = tf.expand_dims(pred_mel, 0)
display_mel(np.exp(pred_mel[0]).T, sr)

# In[]:
init_pred_epoch = 0
last_pred_epoch = 201
# In[]:
_ = melT.predict(pred_mel, MAX_LENGTH=1)
for epoch_n in range(init_pred_epoch, last_pred_epoch + 1, 1):
    print(epoch_n)
    if Path(f'weights/{WEIGHTS_ID}_weights_{epoch_n}.hdf5').exists():
        melT.load_weights(f'weights/{WEIGHTS_ID}_weights_{epoch_n}.hdf5')
        out[epoch_n] = melT.predict_with_target(pred_mel, pred_mel, MAX_LENGTH=100)

# In[]
init_pred_epoch = 0
last_pred_epoch = 94
# In[]
for epoch_n in out.keys():
    print(epoch_n)
    display_mel(np.exp(out[epoch_n]['own'].numpy()[0].T), sr)
# In[]:range(init_pred_epoch, last_pred_epoch + 1, 100)
for epoch_n in out.keys():
    print(epoch_n)
    display_mel(np.exp(out[epoch_n]['train'].numpy()[0].T), sr)
# In[]:
for epoch_n in range(init_pred_epoch, last_pred_epoch + 1, 100):
    print(epoch_n)
    display_mel(np.exp(out[epoch_n]['TE'].numpy()[0].T), sr)
# In[]:

A = pred_mel[:, :-1, :]
B = pred_mel
out1, _, _ = melT.call(B, A, False, enc_padding_mask, combined_mask, dec_padding_mask, apply_conv=True)
out2, _, _ = melT.call(B, A, False, enc_padding_mask, combined_mask, dec_padding_mask)
out2 = melT.out_module.postnet(out2)

print(np.allclose(out1, out2))
