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

losses = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.BinaryCrossentropy()]
loss_coeffs = [0.5, 0.5]
optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
melT.compile(loss=losses, loss_weights=loss_coeffs, optimizer=optimizer)

EPOCHS = 100

train_dataset = []
for i in np.arange(0, norm_ms.shape[0] - 200, 200):
    train_dataset.append([norm_ms[i : i + 100, :], norm_ms[i + 100 : i + 200, :]])
train_dataset = np.array(train_dataset)
print(f'train_dataset.shape: {train_dataset.shape}')
stop_prob = np.zeros((train_dataset.shape[0:3] + (1,)))
stop_prob[:, :, -1] = 1
print(f'stop_prob.shape: {stop_prob.shape}')
print(list(zip(train_dataset, stop_prob))[0][0].shape)
print(list(zip(train_dataset, stop_prob))[0][1].shape)
# print(list(zip(train_dataset, stop_prob))[0][1])

losses = []
for epoch in range(EPOCHS):
    start = time.time()
    for i, (mel, stop) in enumerate(zip(train_dataset, stop_prob)):
        # print(f'step {i}')
        # print('mel.shape:', mel.shape)
        # print('stop.shape', stop.shape)
        gradients, loss, tar_real, predictions, stop_pred = melT.train_step(mel, mel, stop)
        losses.append(loss)
        print('loss:', loss.numpy())

    print('Epoch {} took {} secs\n'.format(epoch, time.time() - start))

out = melT.predict(norm_ms, MAX_LENGTH=100)
print(out['output'].shape)
melT.save_weights('melT_weights.hdf5')
