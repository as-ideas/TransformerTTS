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


power_exp = 1
n_fft = 1024
win_length = 1024
MEL_CHANNELS = 128
y, sr = librosa.load('/Users/fcardina/Downloads/LJ001-0185.wav')
ms = librosa.feature.melspectrogram(
    y=y, sr=sr, n_mels=MEL_CHANNELS, power=power_exp, n_fft=n_fft, win_length=win_length, hop_length=256, fmin=0, fmax=8000
)
norm_ms = np.log(ms.clip(1e-5)).T

print((norm_ms.min(), norm_ms.max()))
start_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) - 1.0
end_vec = np.ones((1, MEL_CHANNELS)) * np.log(1e-5) - 2.0
norm_ms = np.concatenate([start_vec, norm_ms, end_vec])
stop_prob = np.zeros(norm_ms.shape[0])
norm_ms = np.tile(norm_ms, (16, 1, 1))
stop_prob = np.tile(stop_prob, (16, 1))
stop_prob = np.expand_dims(stop_prob, -1)

params = {
    'num_layers': 1,
    'd_model': 128,
    'num_heads': 1,
    'dff': 128,
    'pe_input': 300,
    'pe_target': 300,
    'start_vec': start_vec,
    'mel_channels': MEL_CHANNELS,
    'conv_filters': 64,
    'postnet_conv_layers': 5,
    'postnet_kernel_size': 5,
    'rate': 0.0,
}
melT = MelTransformer(**params)

losses = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.BinaryCrossentropy()]
loss_coeffs = [0.5, 0.5]
optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
melT.compile(loss=losses, loss_weights=loss_coeffs, optimizer=optimizer)
_ = melT.predict(norm_ms[0], MAX_LENGTH=1)
starting_epochs = 1400
melT.load_weights(f'melT_weights_{starting_epochs}.hdf5')

EPOCHS = 1000
losses = []
for epoch in range(EPOCHS):
    start = time.time()
    gradients, loss, tar_real, predictions, stop_pred, loss_vals = melT.train_step(inp=norm_ms, tar=norm_ms, stop_prob=stop_prob)
    print('losses:', loss_vals)
    print('Epoch {} took {} secs\n'.format(epoch, time.time() - start))
melT.save_weights(f'melT_weights_{starting_epochs+EPOCHS}.hdf5')

norm_ms = norm_ms[0]
out = melT.predict(norm_ms, MAX_LENGTH=300, target=norm_ms)
mel_out = out['output'].numpy()
mel_comparison_out = out['target_output'].numpy()
mel_out = np.exp(mel_out).T
mel_comparison_out = np.exp(mel_comparison_out).T

stft = librosa.feature.inverse.mel_to_stft(mel_out, sr=22050, n_fft=n_fft, power=power_exp, fmin=0, fmax=8000)
wav = librosa.feature.inverse.griffinlim(stft, n_iter=60, hop_length=256, win_length=win_length)
librosa.output.write_wav(f'/tmp/sample_{MEL_CHANNELS}_enforce_pred_{starting_epochs+EPOCHS}.wav', wav, sr)

stft = librosa.feature.inverse.mel_to_stft(mel_comparison_out, sr=22050, n_fft=n_fft, power=power_exp, fmin=0, fmax=8000)
wav = librosa.feature.inverse.griffinlim(stft, n_iter=60, hop_length=256, win_length=win_length)
librosa.output.write_wav(f'/tmp/sample_{MEL_CHANNELS}_test_target.wav', wav, sr)
