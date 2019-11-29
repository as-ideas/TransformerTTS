import librosa
import numpy as np
import time
import tensorflow as tf
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).absolute().parent
sys.path.append(SCRIPT_DIR.parent.as_posix())
from src.models import MelTransformer
from utils import display_mel

tf.random.set_seed(10)
np.random.seed(42)
# In[]: VARIABLES

USE_CONV = True
BATCH_SIZE = 32
WAV_FILE = '/Users/fcardina/Downloads/LJ001-0173.wav'
WEIGHTS_ID = 'melT_TAILconv'
MEL_CHANNELS = 128
DROPOUT = 0.0
LEARNING_RATE = 1e-5
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
y, sr = librosa.load(WAV_FILE)
ms = librosa.feature.melspectrogram(
    y=y, sr=sr, n_mels=MEL_CHANNELS, power=power_exp, n_fft=n_fft, win_length=win_length, hop_length=256, fmin=0, fmax=8000
)
norm_ms = np.log(ms.clip(1e-5)).T
# In[]: CREATE DATASET FROM PARTS OF MEL SPECTROGRAM
train_samples = []
for i in range(10):
    cursor = 0
    while cursor < (norm_ms.shape[0] - 100):
        size = np.random.randint(50, 100)
        sample = norm_ms[cursor : cursor + size, :]
        sample = np.concatenate([start_vec, sample, end_vec])
        stop_probs = np.zeros(size + 2)
        stop_probs[-1] = 1
        train_samples.append((sample, stop_probs))
        cursor += size


train_gen = lambda: (mel for mel in train_samples)
train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float64, tf.int64))
train_dataset = train_dataset.padded_batch(16, padded_shapes=([-1, MEL_CHANNELS], [-1]))

# In[]: CREATE THE MODEL
melT = MelTransformer(**params)

# In[]: OPTIMIZATION PARAMETERS
# losses = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.BinaryCrossentropy()]
losses = [tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.BinaryCrossentropy()]
loss_coeffs = [0.5, 0.5]
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
melT.compile(loss=losses, loss_weights=loss_coeffs, optimizer=optimizer)

# In[]: TRAIN THE MODEL
EPOCHS = 2000
starting_epochs = 0
if starting_epochs > 0:
    _ = melT.predict(norm_ms, MAX_LENGTH=1)
    melT.load_weights(f'weights/{WEIGHTS_ID}_weights_{starting_epochs}.hdf5')

for epoch in range(EPOCHS + 1):
    losses = []
    start = time.time()
    for i, (mel, stop) in enumerate(train_dataset):
        gradients, loss, tar_real, predictions, stop_prob, loss_vals = melT.train_step(mel, mel, stop)
        losses.append(loss)

    print('Epoch {} took {} secs. \nAvg loss: {} \n'.format(epoch, time.time() - start, np.mean(losses)))
    if epoch % 100 == 0:
        melT.save_weights(f'weights/{WEIGHTS_ID}_weights_{epoch+starting_epochs}.hdf5')
