import argparse
import os
import random

import librosa
import numpy as np
import tqdm

metadata_file = '/Users/cschaefe/datasets/LJSpeech/LJSpeech-1.1/metadata.csv'
base_path_wavs = '/Users/cschaefe/datasets/LJSpeech/LJSpeech-1.1/wavs'

parser = argparse.ArgumentParser()
parser.add_argument('--sampling_rate', dest='SAMPLING_RATE', default=22050, type=int)
parser.add_argument('--n_fft', dest='N_FFT', default=1024, type=int)
parser.add_argument('--mel_channels', dest='MEL_CHANNELS', default=80, type=int)
parser.add_argument('--f_min', dest='F_MIN', default=0, type=int)
parser.add_argument('--f_max', dest='F_MAX', default=8000, type=int)
parser.add_argument('--win_length', dest='WIN_LENGTH', default=1024, type=int)
parser.add_argument('--hop_length', dest='HOP_LENGTH', default=256, type=int)
parser.add_argument('--random_seed', dest='RANDOM_SEED', default=42, type=int)
parser.add_argument('--test_size', dest='TEST_SIZE', default=100, type=int)
parser.add_argument('--meta_file', dest='META_FILE', type=str, required=True)
parser.add_argument('--wav_path', dest='WAV_PATH', type=str, required=True)
parser.add_argument('--target_path', dest='TARGET_PATH', type=str, required=True)

args = parser.parse_args()
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))

mel_dir = os.path.join(args.TARGET_PATH, 'mels')
if not os.path.exists(mel_dir):
    os.makedirs(mel_dir)

audio_data = []
with open(metadata_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for l in lines:
        l_split = l.split('|')
        filename, text = l_split[0], l_split[-1]
        if filename.endswith('.wav'):
            filename = filename.split('.')[-1]
        audio_data.append((filename, text))

random.seed(args.RANDOM_SEED)
random.shuffle(audio_data)
train_metafile = os.path.join(args.TARGET_PATH, 'train_metafile.txt')
test_metafile = os.path.join(args.TARGET_PATH, 'test_metafile.txt')
test_lines = [''.join([mel_path, '|', text]) for mel_path, text in audio_data[:args.TEST_SIZE]]
train_lines = [''.join([mel_path, '|', text]) for mel_path, text in audio_data[args.TEST_SIZE:-1]]
with open(test_metafile, 'w+', encoding='utf-8') as test_f:
    test_f.writelines(test_lines)
with open(train_metafile, 'w+', encoding='utf-8') as train_f:
    train_f.writelines(train_lines)

for i in tqdm.tqdm(range(len(audio_data))):
    filename, text = audio_data[i]
    wav_path = os.path.join(base_path_wavs, filename + '.wav')
    y, sr = librosa.load(wav_path, sr=args.SAMPLING_RATE)
    S = librosa.feature.melspectrogram(y=y,
                                       sr=args.SAMPLING_RATE,
                                       n_mels=args.MEL_CHANNELS,
                                       power=1,
                                       n_fft=args.N_FFT,
                                       win_length=args.WIN_LENGTH,
                                       hop_length=args.HOP_LENGTH,
                                       fmin=args.F_MIN,
                                       fmax=args.F_MAX)
    S = S.T
    mel_path = os.path.join(mel_dir, filename)
    np.save(mel_path, S)


