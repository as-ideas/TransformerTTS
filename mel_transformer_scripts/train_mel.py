import time
import sys
from pathlib import Path
import argparse

import librosa
import numpy as np
import tensorflow as tf

SCRIPT_DIR = Path(__file__).absolute().parent
sys.path.append(SCRIPT_DIR.parent.as_posix())
from model.models import MelTransformer
from model.transformer_utils import display_mel

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', dest='WAV_DIR', default=SCRIPT_DIR.parent / 'data/LJSpeech-1.1/wavs', type=str)
parser.add_argument('--n_samples', dest='MAX_SAMPLES', default=300, type=int)
parser.add_argument('--dropout', dest='DROPOUT', default=0.1, type=float)
parser.add_argument('--noisestd', dest='NOISE_STD', default=0.2, type=float)
parser.add_argument('--mel', dest='MEL_CHANNELS', default=128, type=int)
parser.add_argument('--split', dest='SPLIT_FILES', action='store_true')
parser.add_argument('--min_size', dest='MIN_SAMPLE_SIZE', default=5, type=int)
parser.add_argument('--max_size', dest='MAX_SAMPLE_SIZE', default=300, type=int)
parser.add_argument('--epochs', dest='EPOCHS', default=3000, type=int)
parser.add_argument('--starting_epoch', dest='starting_epoch', default=0, type=int)
parser.add_argument('--batch_size', dest='BATCH_SIZE', default=128, type=int)
parser.add_argument('--learning_rate', dest='LEARNING_RATE', default=1e-4, type=float)
parser.add_argument('--epoch_without_min', dest='HOW_MANY_EPOCHS_WITHOUT_NEW_MIN', default=10, type=int)
parser.add_argument('--max_lr_halvenings', dest='MAX_LR_HALVENING', default=10, type=int)
parser.add_argument('--pass_per_file', dest='N_TIMES_TO_ITERATE_THROUGH_EACH_SAMPLE', default=1, type=int)
parser.add_argument('--weights_id', dest='WEIGHTS_ID', default='melT_INconvMELlossLEFTpadding', type=str)
parser.add_argument('--weights_dir', dest='WEIGHTS_DIR', default=SCRIPT_DIR.parent / 'weights', type=str)
parser.add_argument('--sample_out_dir', dest='SAMPLE_OUT_DIR', default=SCRIPT_DIR.parent / 'samples', type=str)

args = parser.parse_args()

tf.random.set_seed(10)
np.random.seed(42)

# In[]: VARIABLES
WAV_DIR = Path(args.WAV_DIR)
WEIGHTS_DIR = Path(args.WEIGHTS_DIR)
SAMPLE_OUT_DIR = Path(args.SAMPLE_OUT_DIR)
SAMPLE_OUT_PATH = SAMPLE_OUT_DIR / args.WEIGHTS_ID / 'mel_out'
WEIGHTS_DIR.mkdir(exist_ok=True)
SAMPLE_OUT_PATH.mkdir(exist_ok=True, parents=True)
WEIGHTS_PATH = str(WEIGHTS_DIR / args.WEIGHTS_ID)
start_vec = np.ones((1, args.MEL_CHANNELS)) * np.log(1e-5) - 2.0
end_vec = np.ones((1, args.MEL_CHANNELS)) * np.log(1e-5) + 2.0

params = {
    'num_layers': 4,
    'd_model': 256,
    'num_heads': 2,
    'dff': 256,
    'pe_input': 3000,
    'pe_target': 3000,
    'start_vec': start_vec,
    'mel_channels': args.MEL_CHANNELS,
    'conv_filters': 256,
    'postnet_conv_layers': 3,
    'postnet_kernel_size': 5,
    'rate': args.DROPOUT,
}

losses = [tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.BinaryCrossentropy(),
          tf.keras.losses.MeanAbsoluteError()]
loss_coeffs = [1.0, 1.0, 1.0]
optimizer = tf.keras.optimizers.Adam(args.LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# In[]: GET MEL SPECTROGRAM FROM WAV
power_exp = 1
n_fft = 1024
win_length = 1024


def get_norm_ms(wav_path):
    y, sr = librosa.load(wav_path)
    ms = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=args.MEL_CHANNELS,
        power=power_exp,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=256,
        fmin=0,
        fmax=8000,
    )
    norm_ms = np.log(ms.clip(1e-5)).T
    return norm_ms, sr


# In[]: CREATE DATASET FROM PARTS OF MEL SPECTROGRAM
train_samples = []
wav_file_list = [x for x in WAV_DIR.iterdir() if str(x).endswith('.wav')][: args.MAX_SAMPLES]
SAMPLE_WAV = wav_file_list[0]
sample_norm_mel, sr = get_norm_ms(SAMPLE_WAV)
sample_norm_mel = np.concatenate([start_vec, sample_norm_mel, end_vec])
sample_norm_mel = tf.expand_dims(sample_norm_mel, 0)

for wav_file in wav_file_list:
    norm_ms, sr = get_norm_ms(wav_file)
    for i in range(args.N_TIMES_TO_ITERATE_THROUGH_EACH_SAMPLE):
        cursor = 0
        while cursor <= (norm_ms.shape[0] - args.MAX_SAMPLE_SIZE):
            if not args.SPLIT_FILES:
                size = args.MAX_SAMPLE_SIZE
            else:
                size = np.random.randint(args.MIN_SAMPLE_SIZE, args.MAX_SAMPLE_SIZE)
            sample = norm_ms[cursor: cursor + size, :]
            noise = np.random.randn(*sample.shape) * (args.NOISE_STD) ** 2
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
train_dataset = train_dataset.padded_batch(args.BATCH_SIZE, padded_shapes=([-1, args.MEL_CHANNELS], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# In[]: CREATE THE MODEL
melT = MelTransformer(**params)
melT.compile(loss=losses, loss_weights=loss_coeffs, optimizer=optimizer)

# In[]: TRAIN THE MODEL
if args.starting_epoch > 0:
    _ = melT.predict(tf.expand_dims(norm_ms, 0), max_length=1)
    melT.load_weights(f'{WEIGHTS_PATH}_weights_{args.starting_epoch}.hdf5')

epoch_losses = []
lr_halvenings = 0
min_epoch = 0
for epoch in range(args.EPOCHS + 1):
    losses = []
    start = time.time()
    for i, (mel, stop) in enumerate(train_dataset):
        out = melT.train_step(mel, mel, stop)
        losses.append(out['loss'])
    epoch_losses.append(np.mean(losses))
    print(
        'Epoch {} took {} secs. \nAvg loss: {} \n'.format(args.starting_epoch + epoch, time.time() - start,
                                                          epoch_losses[epoch])
    )
    min_loss = np.min(epoch_losses)  # yeah..
    
    if epoch_losses[epoch] == min_loss:
        min_epoch = epoch
        melT.save_weights(f'{WEIGHTS_PATH}_weights_{epoch + args.starting_epoch}.hdf5')
    if epoch - min_epoch > args.HOW_MANY_EPOCHS_WITHOUT_NEW_MIN:
        if lr_halvenings > args.MAX_LR_HALVENING:
            print(f'Loss has likely stopped improving.\nStopping.')
            break
        print(f'Loss has not improved enough for {epoch - min_epoch} epochs.\nHalvening learning rate.')
        melT.optimizer.learning_rate.assign(melT.optimizer.learning_rate * 0.5)
        lr_halvenings += 1
        min_epoch = epoch
    if epoch_losses[epoch] == min_loss:
        out = melT.predict_with_target(sample_norm_mel, sample_norm_mel, MAX_LENGTH=50)
        for t in ['own', 'TE', 'train']:
            mel_out = np.exp(out[t].numpy()[0].T)
            display_mel(mel_out, sr, file=f'{str(SAMPLE_OUT_PATH)}_{t}_e{args.starting_epoch + epoch}.png')
