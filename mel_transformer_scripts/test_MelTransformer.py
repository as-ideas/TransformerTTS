from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).absolute().parent
sys.path.append(SCRIPT_DIR.parent.as_posix())

# control_values_path = Path(SCRIPT_DIR) / 'MelTransformer_control_values.pkl'
# if not control_values_path.exists():
#     print('MISSING CONTROL VALUES')
#     print(f'First create control values by running \n{SCRIPT_DIR}/get_control_value.py')
#     exit()

import time
import numpy as np
import tensorflow as tf
from model.models import MelTransformer

np.random.seed(42)
tf.random.set_seed(42)

SEQ_LEN1 = 60
SEQ_LEN2 = 30
MEL_CHANNELS = 80
BATCH_SIZE = 3
EPOCHS = 2
LEARNING_RATE = 1e-5
start_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) - 2.0
end_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) + 2.0

TEST_MELS = [np.random.random((SEQ_LEN1, MEL_CHANNELS)) * 5 - 4.0,
             np.random.random((SEQ_LEN2, MEL_CHANNELS)) * 2 - 6.0] * 5
STOP_PROBS = [np.zeros(SEQ_LEN1 + 2), np.zeros(SEQ_LEN2 + 2)] * 5
train_samples = []
for mel, stop in zip(TEST_MELS, STOP_PROBS):
    mel = np.concatenate([start_vec, mel, end_vec])
    stop[-1] = 1
    train_samples.append((mel, stop))

params = {
    'num_layers': 2,
    'd_model': 64,
    'num_heads': 2,
    'dff': 32,
    'pe_input': 61,
    'pe_target': 61,
    'start_vec': start_vec,
    'mel_channels': MEL_CHANNELS,
    'conv_filters': 32,
    'postnet_conv_layers': 2,
    'postnet_kernel_size': 5,
    'rate': 0.1,
}

train_gen = lambda: (mel for mel in train_samples)
train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float64, tf.int64))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes=([-1, MEL_CHANNELS], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

losses = [tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.BinaryCrossentropy(),
          tf.keras.losses.MeanAbsoluteError()]
loss_coeffs = [1.0, 1.0, 1.0]
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

melT = MelTransformer(**params)
melT.compile(loss=losses, loss_weights=loss_coeffs, optimizer=optimizer)

epoch_losses = []
for epoch in range(EPOCHS):
    losses = []
    start = time.time()
    for i, (mel, stop) in enumerate(train_dataset):
        out = melT.train_step(mel, mel, stop)
        losses.append(out['loss'])
    epoch_losses.append(np.mean(losses))
    print('Epoch {} took {} secs. \nAvg loss: {} \n'.format(epoch, time.time() - start, epoch_losses[epoch]))
    
    if epoch_losses[epoch] == min_loss:
        out = melT.predict(sample_norm_mel, max_length=50)
        print('mel', out['mel'])
        print('attn', out['attn'])
