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

out = melT.predict(norm_ms, MAX_LENGTH=1)
melT.load_weights('melT_weights.hdf5')
out = melT.predict(norm_ms, MAX_LENGTH=100)
print(out['output'].shape)
display_mel(out['output'].numpy().T, sr)
