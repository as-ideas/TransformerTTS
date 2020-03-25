import argparse
import os
import random

import librosa
import numpy as np
import tqdm

from preprocessing.text_processing import Phonemizer, TextCleaner
parser = argparse.ArgumentParser()
parser.add_argument('--sampling_rate', dest='SAMPLING_RATE', default=22050, type=int)
parser.add_argument('--n_fft', dest='N_FFT', default=1024, type=int)
parser.add_argument('--mel_channels', dest='MEL_CHANNELS', default=80, type=int)
parser.add_argument('--f_min', dest='F_MIN', default=0, type=int)
parser.add_argument('--f_max', dest='F_MAX', default=8000, type=int)
parser.add_argument('--win_length', dest='WIN_LENGTH', default=1024, type=int)
parser.add_argument('--hop_length', dest='HOP_LENGTH', default=256, type=int)
parser.add_argument('--random_seed', dest='RANDOM_SEED', default=42, type=int)
parser.add_argument('--no_phonemes', dest='PHONEMIZE', action='store_false')
parser.add_argument('--phoneme_language', dest='LANGUAGE', default='', type=str)
parser.add_argument('--test_size', dest='TEST_SIZE', default=100, type=int)
parser.add_argument('--meta_file', dest='META_FILE', type=str, required=True)
parser.add_argument('--wav_dir', dest='WAV_DIR', type=str, required=True)
parser.add_argument('--target_dir', dest='TARGET_DIR', type=str, required=True)
parser.add_argument('--njobs', dest='NJOBS', type=int, default=16)
parser.add_argument('--ph_splits', dest='PH_SPLITS', type=int, default=250)

args = parser.parse_args()
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))
if args.PHONEMIZE:
    phonemizer = Phonemizer(args.LANGUAGE)
    
mel_dir = os.path.join(args.TARGET_DIR, 'mels')
if not os.path.exists(mel_dir):
    os.makedirs(mel_dir)
print('\nLoading and cleaning text')
text_cleaner = TextCleaner()
audio_data = []
with open(args.META_FILE, 'r', encoding='utf-8') as f:
    for l in f.readlines():
        l_split = l.split('|')
        filename, text = l_split[0], l_split[-1]
        if filename.endswith('.wav'):
            filename = filename.split('.')[-1]
        text = text_cleaner.clean(text)
        audio_data.append((filename, text))

if args.PHONEMIZE:
    print('\nPhonemizing')
    audio_data = np.array(audio_data)
    texts = audio_data[:, 1]
    split = args.PH_SPLITS
    phonemes = []
    for i in range(0, len(audio_data), split):
        part = texts[i: i + split]
        part = phonemizer.encode(part, njobs=args.NJOBS)
        phonemes.extend(part)
    audio_data = np.concatenate([np.array(audio_data), np.expand_dims(phonemes, axis=1)], axis=1)
print('\nBuilding dataset and writing files')
random.seed(args.RANDOM_SEED)
random.shuffle(audio_data)
test_metafile = os.path.join(args.TARGET_DIR, 'test_metafile.txt')
train_metafile = os.path.join(args.TARGET_DIR, 'train_metafile.txt')

if args.PHONEMIZE:
    test_lines = [''.join([filename, '|', text, '|', phon, '\n']) for filename, text, phon in audio_data[:args.TEST_SIZE]]
    train_lines = [''.join([filename, '|', text, '|', phon, '\n']) for filename, text, phon in audio_data[args.TEST_SIZE:-1]]
else:
    test_lines = [''.join([filename, '|', text, '\n']) for filename, text in audio_data[:args.TEST_SIZE]]
    train_lines = [''.join([filename, '|', text, '\n']) for filename, text in audio_data[args.TEST_SIZE:-1]]
    
with open(test_metafile, 'w+', encoding='utf-8') as test_f:
    test_f.writelines(test_lines)
with open(train_metafile, 'w+', encoding='utf-8') as train_f:
    train_f.writelines(train_lines)

for i in tqdm.tqdm(range(len(audio_data))):
    filename, _, _ = audio_data[i]
    wav_path = os.path.join(args.WAV_DIR, filename + '.wav')
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
    mel_path = os.path.join(mel_dir, filename)
    np.save(mel_path, S.T)
print('\nDone')