import argparse
from pathlib import Path

import numpy as np
import tqdm

from preprocessing.text import Pipeline
from preprocessing.datasets import MetadataReader
from utils.config_manager import Config
from utils.audio import Audio

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--skip_phonemes', action='store_true')
parser.add_argument('--skip_mels', action='store_true')
parser.add_argument('--skip_wavs', action='store_true')
parser.add_argument('--phonemizer_parallel_jobs', type=int, default=16)
parser.add_argument('--phonemizer_batch_size', type=int, default=16)

args = parser.parse_args()
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))

cm = Config(args.config, model_kind='autoregressive')
cm.create_remove_dirs()
metadatareader = MetadataReader.default_original_from_config(cm, metadata_reading_function=None)
np_data = np.array(metadatareader.data)
texts = np_data[:, 1]
filenames = np_data[:, 0]

if not args.skip_phonemes:
    phonemized_metadata_path = Path(cm.train_datadir) / 'phonemized_metadata.txt'
    train_metadata_path = Path(cm.train_datadir) / cm.config['train_metadata_filename']
    test_metadata_path = Path(cm.train_datadir) / cm.config['valid_metadata_filename']
    metadata_len = len(metadatareader.data)
    test_len = cm.config['n_test']
    train_len = metadata_len - test_len
    print(f'\nReading metadata from {metadatareader.metadata_path}')
    print(f'\nMetadata contains {metadata_len} lines.')
    print(f'Files will be stored under {cm.train_datadir}')
    print(f' - all: {phonemized_metadata_path}')
    print(f' - {train_len} training lines: {train_metadata_path}')
    print(f' - {test_len} validation lines: {test_metadata_path}')
    
    print('\nMetadata head:')
    for i in range(5):
        print(f'{metadatareader.data[i]}')
    print('\nMetadata tail:')
    for i in range(1, 5):
        print(f'{metadatareader.data[-i]}')
    
    # run cleaner on raw text
    text_proc = Pipeline.default_training_pipeline(cm.config['phoneme_language'], add_start_end=False,
                                                   with_stress=cm.config['with_stress'])
    clean_texts = text_proc.cleaner(list(texts))
    print('\nCleaned metadata head:')
    for i in range(5):
        print(f'{(filenames[i], clean_texts[i])}')
    print('\nCleaned metadata tail:')
    for i in range(1, 5):
        print(f'{(filenames[-i], clean_texts[-i])}')
    
    print('\nPHONEMIZING')
    phonemes = []
    for i in tqdm.tqdm(range(0, len(clean_texts), args.phonemizer_batch_size)):
        batch = clean_texts[i: i + args.phonemizer_batch_size]
        batch = text_proc.phonemizer(batch, njobs=args.phonemizer_parallel_jobs)
        phonemes.extend(batch)
    
    new_metadata = [''.join([fname, '|', ph, '\n']) for fname, ph in zip(filenames, phonemes)]
    shuffled_metadata = np.random.permutation(new_metadata)
    train_metadata = shuffled_metadata[0:train_len]
    test_metadata = shuffled_metadata[:test_len]
    
    # some checks
    assert metadata_len == len(phonemes), \
        f'Length of metadata ({metadata_len}) does not match the length of the phoneme array ({len(phonemes)}). Check for empty text lines in metadata.'
    assert len(train_metadata) + len(test_metadata) == metadata_len, \
        'Train and/or validation lengths incorrect.'
    
    print('\nPhonemized metadata head:')
    for i in range(5):
        print(f'{new_metadata[i]}')
    print('\nPhonemized metadata tail:')
    for i in range(1, 5):
        print(f'{new_metadata[-i]}')
    
    with open(phonemized_metadata_path, 'w+', encoding='utf-8') as file:
        file.writelines(new_metadata)
    with open(train_metadata_path, 'w+', encoding='utf-8') as file:
        file.writelines(train_metadata)
    with open(test_metadata_path, 'w+', encoding='utf-8') as file:
        file.writelines(test_metadata)

if (not args.skip_mels) or (not args.skip_wavs):
    print(f"\nMels and resampled wavs will be respectibvely stored under")
    print(f"{cm.train_datadir / 'mels'} and {cm.train_datadir / 'resampled_wavs'}")
    print(f'RESAMPLING WAVS and COMPUTING MELS')
    (cm.train_datadir / 'resampled_wavs').mkdir(exist_ok=True)
    (cm.train_datadir / 'mels').mkdir(exist_ok=True)
    audio = Audio(config=cm.config)
    for i in tqdm.tqdm(range(len(filenames))):
        wav_path = (metadatareader.wav_directory / filenames[i]).with_suffix('.wav')
        y, sr = audio.load_wav(str(wav_path))
        if not args.skip_mels:
            mel = audio.mel_spectrogram(y)
            assert mel.shape[1] == audio.config['mel_channels']
            mel_path = (cm.train_datadir / 'mels' / filenames[i]).with_suffix('.npy')
            np.save(mel_path, mel)
            # TODO: switch to padding = -1 (or wav pad value)
            y = np.pad(y, (0, cm.config['n_fft']), constant_values=cm.config['wav_padding_value'])
            # y = np.pad(y, (0, cm.config['n_fft']), mode='edge')
            y = y [:mel.shape[0]*audio.config['hop_length']]
            assert len(y) == mel.shape[0]*audio.config['hop_length']
            wav_path = (cm.train_datadir / 'resampled_wavs' / filenames[i]).with_suffix('.npy')
            np.save(wav_path, y)
print('\nDone')
