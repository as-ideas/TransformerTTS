import os
import random

import numpy as np
import time
import tensorflow as tf
from pathlib import Path
import argparse

from model.transformer_factory import new_mel_transformer
from utils import display_mel

parser = argparse.ArgumentParser()
parser.add_argument('--mel_dir', dest='MEL_DIR', type=str, required=True)
parser.add_argument('--n_samples', dest='N_SAMPLES', default=5000, type=int)
parser.add_argument('--dropout', dest='DROPOUT', default=0.1, type=float)
parser.add_argument('--noisestd', dest='NOISE_STD', default=0.2, type=float)
parser.add_argument('--mel_channels', dest='MEL_CHANNELS', default=80, type=int)
parser.add_argument('--epochs', dest='EPOCHS', default=3000, type=int)
parser.add_argument('--starting_epoch', dest='starting_epoch', default=0, type=int)
parser.add_argument('--batch_size', dest='BATCH_SIZE', default=8, type=int)
parser.add_argument('--learning_rate', dest='LEARNING_RATE', default=1e-4, type=float)
parser.add_argument('--sampling_rate', dest='SAMPLING_RATE', default=22050, type=int)
parser.add_argument('--weights_id', dest='WEIGHTS_ID', default='mel_transformer_INconvMELlossLEFTpadding', type=str)
parser.add_argument('--weights_dir', dest='WEIGHTS_DIR', default='/tmp/weights', type=str)
parser.add_argument('--sample_out_dir', dest='SAMPLE_OUT_DIR', default='/tmp/samples', type=str)
parser.add_argument('--random_seed', dest='RANDOM_SEED', default=42, type=int)

args = parser.parse_args()

tf.random.set_seed(args.RANDOM_SEED)
np.random.seed(args.RANDOM_SEED)

MEL_DIR = Path(args.MEL_DIR)
WEIGHTS_DIR = Path(args.WEIGHTS_DIR)
SAMPLE_OUT_DIR = Path(args.SAMPLE_OUT_DIR)
SAMPLE_OUT_PATH = SAMPLE_OUT_DIR / args.WEIGHTS_ID / 'mel_out'
WEIGHTS_DIR.mkdir(exist_ok=True)
SAMPLE_OUT_PATH.mkdir(exist_ok=True, parents=True)
WEIGHTS_PATH = str(WEIGHTS_DIR / args.WEIGHTS_ID)
start_vec = np.ones((1, args.MEL_CHANNELS)) * -3
end_vec = np.ones((1, args.MEL_CHANNELS)) * -6

losses = [tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.MeanAbsoluteError()]
loss_coeffs = [1.0, 1.0, 1.0]
optimizer = tf.keras.optimizers.Adam(args.LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def get_norm_mel(mel_path):
    mel = np.load(mel_path)
    norm_mel = np.log(mel.clip(1e-5))
    norm_mel = np.concatenate([start_vec, norm_mel, end_vec])
    return norm_mel


train_data = []
mel_files = [os.path.join(MEL_DIR, f) for f in os.listdir(MEL_DIR)]
mel_files.sort()
random.seed(args.RANDOM_SEED)
random.shuffle(mel_files)
for mel_file in mel_files[:args.N_SAMPLES]:
    norm_mel = get_norm_mel(mel_file)
    stop_probs = np.zeros(norm_mel.shape[0], dtype=np.int64)
    train_data.append((norm_mel, stop_probs))

sample_norm_mel = tf.expand_dims(train_data[0][0], 0)
train_dataset = tf.data.Dataset.from_generator(lambda: train_data, output_types=(tf.float64, tf.int64))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.padded_batch(args.BATCH_SIZE, padded_shapes=([-1, args.MEL_CHANNELS], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

mel_transformer = new_mel_transformer(start_vec=start_vec,
                                      num_layers=2,
                                      d_model=64,
                                      num_heads=2,
                                      dff=32,
                                      max_position_encoding=1000,
                                      dropout_rate=0.1,
                                      mel_channels=80,
                                      postnet_conv_filters=32,
                                      postnet_conv_layers=2,
                                      postnet_kernel_size=5)

mel_transformer.compile(loss=losses, loss_weights=loss_coeffs, optimizer=optimizer)
epoch_losses = []
lr_halvenings = 0
min_epoch = 0
for epoch in range(args.EPOCHS + 1):
    losses = []
    start = time.time()
    for i, (mel, stop) in enumerate(train_dataset):
        out = mel_transformer.train_step(mel, mel, stop)
        losses.append(out['loss'])
        print('{} {}'.format(i, out['loss']))
    epoch_losses.append(np.mean(losses))
    print(
        'Epoch {} took {} secs. \nAvg loss: {} \n'.format(args.starting_epoch + epoch, time.time() - start, epoch_losses[epoch])
    )
    min_loss = np.min(epoch_losses)  # yeah..

    if epoch_losses[epoch] == min_loss:
        min_epoch = epoch
        mel_transformer.save_weights(f'{WEIGHTS_PATH}_weights_{epoch+args.starting_epoch}.hdf5')
    if epoch_losses[epoch] == min_loss:
        out = mel_transformer.predict_with_target(sample_norm_mel, sample_norm_mel, max_length=50)
        for t in ['own', 'TE', 'train']:
            mel_out = np.exp(out[t].numpy()[0].T)
            display_mel(mel_out, args.SAMPLING_RATE, file=f'{str(SAMPLE_OUT_PATH)}_{t}_e{args.starting_epoch+epoch}.png')
