import argparse
import os
import random

import librosa
import numpy as np
import tqdm
import ruamel.yaml

from preprocessing.text_processing import Phonemizer, TextCleaner

parser = argparse.ArgumentParser()
parser.add_argument('--meta_file', dest='META_FILE', type=str, required=True)
parser.add_argument('--wav_dir', dest='WAV_DIR', type=str, required=True)
parser.add_argument('--target_dir', dest='TARGET_DIR', type=str, required=True)
parser.add_argument('--config', dest='CONFIG', type=str, required=True)
parser.add_argument('--njobs', dest='NJOBS', type=int, default=16)
parser.add_argument('--col_sep', dest='COLUMN_SEP', type=str, default='|')
args = parser.parse_args()
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))

yaml = ruamel.yaml.YAML()
config = yaml.load(open(args.CONFIG, 'r'))
phonemizer = Phonemizer(config['phoneme_language'])

mel_dir = os.path.join(args.TARGET_DIR, 'mels')
if not os.path.exists(mel_dir):
    os.makedirs(mel_dir)
print('\nLoading and cleaning text')
text_cleaner = TextCleaner()
audio_data = []
with open(args.META_FILE, 'r', encoding='utf-8') as f:
    for l in f.readlines():
        l_split = l.split(args.COLUMN_SEP)
        filename, text = l_split[0], l_split[-1]
        if filename.endswith('.wav'):
            filename = filename.split('.')[-1]
        text = text_cleaner.clean(text)
        audio_data.append((filename, text))

print('\nPhonemizing')
audio_data = np.array(audio_data)
texts = audio_data[:, 1]
batch_size = 250  # batch phonemization to avoid memory issues.
phonemes = []
for i in tqdm.tqdm(range(0, len(audio_data), batch_size)):
    batch = texts[i: i + batch_size]
    batch = phonemizer.encode(batch, njobs=args.NJOBS, clean=False)
    phonemes.extend(batch)
audio_data = np.concatenate([np.array(audio_data), np.expand_dims(phonemes, axis=1)], axis=1)

print('\nBuilding dataset and writing files')
random.seed(42)
random.shuffle(audio_data)
test_metafile = os.path.join(args.TARGET_DIR, 'test_metafile.txt')
train_metafile = os.path.join(args.TARGET_DIR, 'train_metafile.txt')

test_lines = [''.join([filename, '|', text, '|', phon, '\n']) for filename, text, phon in
              audio_data[:config['n_test']]]
train_lines = [''.join([filename, '|', text, '|', phon, '\n']) for filename, text, phon in
               audio_data[config['n_test']:-1]]

with open(test_metafile, 'w+', encoding='utf-8') as test_f:
    test_f.writelines(test_lines)
with open(train_metafile, 'w+', encoding='utf-8') as train_f:
    train_f.writelines(train_lines)

for i in tqdm.tqdm(range(len(audio_data))):
    filename, _, _ = audio_data[i]
    wav_path = os.path.join(args.WAV_DIR, filename + '.wav')
    y, sr = librosa.load(wav_path, sr=config['sampling_rate'])
    S = librosa.feature.melspectrogram(y=y,
                                       sr=config['sampling_rate'],
                                       n_mels=config['mel_channels'],
                                       power=1,
                                       n_fft=config['n_fft'],
                                       win_length=config['win_lenght'],
                                       hop_length=config['hop_length'],
                                       fmin=config['fmin'],
                                       fmax=config['fmax'])
    mel_path = os.path.join(mel_dir, filename)
    np.save(mel_path, S.T)
print('\nDone')
