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

# WAV_DIR = Path('/Users/fcardina/forge/text-to-speech/data/wav_files/')
WAV_DIR = Path('/Users/fcardina/forge/text-to-speech/data/LJSpeech-1.1/wavs')
MEL_CHANNELS = 128
MAX_SAMPLES = 300
start_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) - 2.0
end_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) + 2.0

WEIGHTS_ID = 'melT_TAILconv'
DROPOUT = 0.05

NOISE_STD = 0.3
N_TIMES_TO_ITERATE_THROUGH_EACH_SAMPLE = 1
SPLIT_FILES = True
if SPLIT_FILES:
    MIN_SAMPLE_SIZE = 5
MAX_SAMPLE_SIZE = 300
# MAX_SAMPLE_SIZE = -1

EPOCHS = 3000
starting_epoch = 0
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
HOW_MANY_EPOCHS_WITHOUT_NEW_MIN = 5
MAX_LR_HALVENING = 10


params = {
    'num_layers': 1,
    'd_model': 256,
    'num_heads': 1,
    'dff': 256,
    'pe_input': 3000,
    'pe_target': 3000,
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


def get_norm_ms(wav_path):
    y, sr = librosa.load(wav_path)
    ms = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=MEL_CHANNELS, power=power_exp, n_fft=n_fft, win_length=win_length, hop_length=256, fmin=0, fmax=8000
    )
    norm_ms = np.log(ms.clip(1e-5)).T
    return norm_ms, sr


# In[]: CREATE DATASET FROM PARTS OF MEL SPECTROGRAM
train_samples = []
wav_file_list = [x for x in WAV_DIR.iterdir() if str(x).endswith('.wav')][:MAX_SAMPLES]
for wav_file in wav_file_list:
    norm_ms, sr = get_norm_ms(wav_file)
    for i in range(N_TIMES_TO_ITERATE_THROUGH_EACH_SAMPLE):
        cursor = 0
        while cursor <= (norm_ms.shape[0] - MAX_SAMPLE_SIZE):
            if not SPLIT_FILES:
                size = MAX_SAMPLE_SIZE
            else:
                size = np.random.randint(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE)
            sample = norm_ms[cursor : cursor + size, :]
            noise = np.random.randn(*sample.shape) * (NOISE_STD) ** 2
            sample += noise
            sample = np.concatenate([start_vec, sample, end_vec])
            stop_probs = np.zeros(size + 2)
            stop_probs[-1] = 1
            train_samples.append((sample, stop_probs))
            cursor += size


print(f'{len(train_samples)} train samples.')
train_gen = lambda: (mel for mel in train_samples)
train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float64, tf.int64))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes=([-1, MEL_CHANNELS], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

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
    print('Epoch {} took {} secs. \nAvg loss: {} \n'.format(starting_epoch + epoch, time.time() - start, epoch_losses[epoch]))
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
