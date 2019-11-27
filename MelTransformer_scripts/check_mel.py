import librosa
import librosa.display
import numpy as np
import time
import tensorflow as tf

# TODO: we are now using transposed mel spectrograms. this needs to be fixed.
#       we are not using start and end vectors. are they needed?
# import IPython.display as ipd

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).absolute().parent
sys.path.append(SCRIPT_DIR.parent.as_posix())
from src.models import MelTransformer
from utils import display_mel

tf.random.set_seed(10)
np.random.seed(42)
# load audio
# get mel

# MAX_POW = 3000

import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa.display import specshow

power_exp = 1
n_fft = 1024
win_length = 1024
MEL_CHANNELS = 128
y, sr = librosa.load('/Users/fcardina/Downloads/LJ001-0185.wav')
# ms = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=MEL_CHANNELS)
# y, ind = librosa.effects.trim(y, top_db=40, frame_length=2048, hop_length=512)
ms = librosa.feature.melspectrogram(
    y=y, sr=sr, n_mels=MEL_CHANNELS, power=power_exp, n_fft=n_fft, win_length=win_length, hop_length=256, fmin=0, fmax=8000
)


norm_ms = np.log(ms.clip(1e-5)).T
print('norm_ms.shape', norm_ms.shape)

print((norm_ms.min(), norm_ms.max()))
start_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) - 1
end_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) - 2
norm_ms = np.concatenate([start_vec, norm_ms, end_vec])
stop_prob = np.zeros(norm_ms.shape[0])
norm_ms = np.tile(norm_ms, (16, 1, 1))
stop_prob = np.tile(stop_prob, (16, 1))
stop_prob = np.expand_dims(stop_prob, -1)
# norm_ms = np.expand_dims(norm_ms, 0)
print('norm_ms.shape', norm_ms.shape)
print('stop_prob.shape', stop_prob.shape)
params = {
    'num_layers': 1,
    'd_model': 256,
    'num_heads': 2,
    'dff': 256,
    'pe_input': 10000,
    'pe_target': 10000,
    'start_vec': start_vec,
    'mel_channels': MEL_CHANNELS,
    'conv_filters': 256,
    'postnet_conv_layers': 5,
    'postnet_kernel_size': 5,
    'rate': 0.1,
}
melT = MelTransformer(**params)

losses = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.BinaryCrossentropy()]
loss_coeffs = [0.5, 0.5]
optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
melT.compile(loss=losses, loss_weights=loss_coeffs, optimizer=optimizer)


EPOCHS = 0
losses = []
for epoch in range(EPOCHS):
    start = time.time()
    gradients, loss, tar_real, predictions, stop_pred = melT.train_step(norm_ms, norm_ms, stop_prob)
    print(stop_pred)
    print('loss:', loss.numpy())
    print('Epoch {} took {} secs\n'.format(epoch, time.time() - start))

norm_ms = norm_ms[0]
print('norm_ms.shape', norm_ms.shape)
# out = melT.predict(norm_ms, MAX_LENGTH=300)
# print(out['output'].shape)
# mel_out = out['output'].numpy().T
# melT.save_weights('melT_weights.hdf5')
mel_out = np.exp(norm_ms).T
stft = librosa.feature.inverse.mel_to_stft(mel_out, sr=22050, n_fft=n_fft, power=power_exp, fmin=0, fmax=8000)
wav = librosa.feature.inverse.griffinlim(stft, n_iter=60, hop_length=256, win_length=win_length)
"""
wav = librosa.feature.inverse.mel_to_audio(S,
                                           sr=sr,
                                           hop_length=256,
                                           n_fft=n_fft,
                                           win_length=win_length,
                                           power=power_exp)
"""
librosa.output.write_wav(f'/tmp/sample_{MEL_CHANNELS}_origin.wav', wav, sr)
