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
y, sr = librosa.load(librosa.util.example_audio_file())
# get mel
MEL_CHANNELS = 128
ms = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=MEL_CHANNELS)
# display_mel(ms, sr)
norm_ms = (ms - ms.min()) / ms.max()
norm_ms = norm_ms.T
print((norm_ms.min(), norm_ms.max()))
print(norm_ms.shape)
# display_mel(norm_ms, sr)

params = {
    'num_layers': 1,
    'd_model': 40,
    'num_heads': 2,
    'dff': 30,
    'pe_input': ms.shape[1] + 1,
    'pe_target': ms.shape[1] + 1,
    'start_vec': np.ones(MEL_CHANNELS) * -1,
    'mel_channels': MEL_CHANNELS,
    'conv_filters': 64,
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
