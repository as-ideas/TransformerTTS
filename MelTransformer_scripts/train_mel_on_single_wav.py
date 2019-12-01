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
EPOCHS = 3000
starting_epoch = 1408
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WAV_FILE = '/Users/fcardina/Downloads/LJ001-0173.wav'
WAV_FILE = '/home/francesco/Downloads/temp/LJ037-0171.wav'
WEIGHTS_ID = 'melT_TAILconv'
MEL_CHANNELS = 128
DROPOUT = 0.05
start_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) - 2.0
end_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) + 2.0
N_TIMES_TO_ITERATE_THROUGH_SAMPLE_TO_CREATE_SAMPLES = 30
MIN_SAMPLE_SIZE = 30
MAX_SAMPLE_SIZE = 100
HOW_MANY_EPOCHS_WITHOUT_NEW_MIN = 5
MAX_LR_HALVENING = 10

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

losses = [tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.BinaryCrossentropy()]
loss_coeffs = [1.0, 1.0]
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# In[]: GET MEL SPECTROGRAM FROM WAV
power_exp = 1
n_fft = 1024
win_length = 1024
y, sr = librosa.load(WAV_FILE)
ms = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=MEL_CHANNELS,
    power=power_exp,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=256,
    fmin=0,
    fmax=8000,
)
norm_ms = np.log(ms.clip(1e-5)).T

# In[]: CREATE DATASET FROM PARTS OF MEL SPECTROGRAM
train_samples = []
for i in range(N_TIMES_TO_ITERATE_THROUGH_SAMPLE_TO_CREATE_SAMPLES):
    cursor = 0
    while cursor < (norm_ms.shape[0] - MAX_SAMPLE_SIZE):
        size = np.random.randint(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE)
        sample = norm_ms[cursor : cursor + size, :]
        sample = np.concatenate([start_vec, sample, end_vec])
        stop_probs = np.zeros(size + 2)
        stop_probs[-1] = 1
        train_samples.append((sample, stop_probs))
        cursor += size
train_gen = lambda: (mel for mel in train_samples)
train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float64, tf.int64))
train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes=([-1, MEL_CHANNELS], [-1]))

# In[]: CREATE THE MODEL
melT = MelTransformer(**params)
melT.compile(loss=losses, loss_weights=loss_coeffs, optimizer=optimizer)

# In[]: TRAIN THE MODEL
if starting_epoch > 0:
    _ = melT.predict(tf.expand_dims(norm_ms, 0), MAX_LENGTH=1)
    melT.load_weights(f'weights/{WEIGHTS_ID}_weights_{starting_epoch}.hdf5')

epoch_losses = []
lr_halvenings = 0
min_epoch = 0
for epoch in range(EPOCHS + 1):
    losses = []
    start = time.time()
    for i, (mel, stop) in enumerate(train_dataset):
        gradients, loss, tar_real, predictions, stop_prob, loss_vals = melT.train_step(mel, mel, stop)
        losses.append(loss)
    epoch_losses.append(np.mean(losses))
    print(
        'Epoch {} took {} secs. \nAvg loss: {} \n'.format(
            starting_epoch + epoch, time.time() - start, epoch_losses[epoch]
        )
    )
    min_loss = np.min(epoch_losses)  # yeah..

    if epoch_losses[epoch] == min_loss:
        min_epoch = epoch
        melT.save_weights(f'weights/{WEIGHTS_ID}_weights_{epoch+starting_epoch}.hdf5')
    if epoch - min_epoch > HOW_MANY_EPOCHS_WITHOUT_NEW_MIN:
        if lr_halvenings > MAX_LR_HALVENING:
            print(f'Loss has likely stopped improving.\nStopping.')
            break
        print(f'Loss has not improved enough for {epoch - min_epoch} epochs.\nHalvening learning rate.')
        melT.optimizer.learning_rate.assign(melT.optimizer.learning_rate * 0.5)
        lr_halvenings += 1
        min_epoch = epoch
