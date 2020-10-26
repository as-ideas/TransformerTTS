import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pickle

import numpy as np
import tqdm

from preprocessing.text import TextToTokens
from preprocessing.datasets import DataReader
from utils.config_manager import Config
from utils.audio import Audio

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--skip_phonemes', action='store_true')
parser.add_argument('--skip_mels', action='store_true')
parser.add_argument('--clean_dataset', action='store_true')
parser.add_argument('--phonemizer_parallel_jobs', type=int, default=16)
parser.add_argument('--phonemizer_batch_size', type=int, default=16)

args = parser.parse_args()
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))

cm = Config(args.config, model_kind='autoregressive')
cm.create_remove_dirs()
metadatareader = DataReader.from_config(cm, kind='original', scan_wavs=True)

if not args.skip_mels:
    import sys
    
    
    def process_wav(wav_path: Path):
        file_name = wav_path.stem
        y, sr = audio.load_wav(str(wav_path))
        mel = audio.mel_spectrogram(y)
        assert mel.shape[1] == audio.config['mel_channels'], len(mel.shape) == 2
        mel_path = (cm.mel_dir / file_name).with_suffix('.npy')
        np.save(mel_path, mel)
        return (file_name, mel.shape[0])
    
    
    def progbar(i, n, size=16):
        done = (i * size) // n
        bar = ''
        for i in range(size):
            bar += '█' if i <= done else '░'
        return bar
    
    
    print(f"Creating mels from all wavs found in {metadatareader.data_directory}")
    print(f"\nMels will be stored stored under")
    print(f"{cm.mel_dir}")
    (cm.mel_dir).mkdir(exist_ok=True)
    audio = Audio(config=cm.config)
    pool = Pool(processes=cpu_count() - 1)
    wav_files = [metadatareader.wav_paths[k] for k in metadatareader.wav_paths]
    len_dict = {}
    remove_files = []
    for i, (fname, mel_len) in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
        bar = progbar(i, len(wav_files))
        message = f'{bar} {i}/{len(wav_files)} '
        sys.stdout.write(f"\r{message}")
        len_dict.update({fname: mel_len})
        if mel_len > cm.config['max_mel_len'] or mel_len < cm.config['min_mel_len']:
            remove_files.append(fname)
    pickle.dump(len_dict, open(cm.data_dir / 'mel_len.pkl', 'wb'))
    pickle.dump(remove_files, open(cm.data_dir / 'under-over_sized_mels.pkl', 'wb'))

if not args.skip_phonemes:
    remove_files = pickle.load(open(cm.data_dir / 'under-over_sized_mels.pkl', 'rb'))
    phonemized_metadata_path = Path(cm.data_dir) / 'phonemized_metadata.txt'
    train_metadata_path = Path(cm.data_dir) / cm.config['train_metadata_filename']
    test_metadata_path = Path(cm.data_dir) / cm.config['valid_metadata_filename']
    print(f'\nReading metadata from {metadatareader.metadata_path}')
    print(f'\nFound {len(metadatareader.filenames)} lines.')
    from preprocessing.text.symbols import _alphabet
    filter_metadata = []
    for fname in metadatareader.filenames:
        item = metadatareader.text_dict[fname]
        non_p = [c for c in item if c in _alphabet]
        if len(non_p) < 1:
            filter_metadata.append(fname)
    if len(filter_metadata) > 0:
        print(f'Removing {len(filter_metadata)} suspiciously short line(s):')
        for fname in filter_metadata:
            print(f'{fname}: {metadatareader.text_dict[fname]}')
    print(f'\nRemoving {len(remove_files)} line(s) due to mel filtering.')
    remove_files += filter_metadata
    metadatareader.filenames = [fname for fname in metadatareader.filenames if fname not in remove_files]
    metadata_len = len(metadatareader.filenames)
    sample_items = np.random.choice(metadatareader.filenames, 5)
    test_len = cm.config['n_test']
    train_len = metadata_len - test_len
    print(f'\nMetadata contains {metadata_len} lines.')
    print(f'\nFiles will be stored under {cm.data_dir}')
    print(f' - all: {phonemized_metadata_path}')
    print(f' - {train_len} training lines: {train_metadata_path}')
    print(f' - {test_len} validation lines: {test_metadata_path}')
    
    print('\nMetadata samples:')
    for i in sample_items:
        print(f'{i}:{metadatareader.text_dict[i]}')
    
    # run cleaner on raw text
    text_proc = TextToTokens.default_training(cm.config['phoneme_language'], add_start_end=False,
                                              with_stress=cm.config['with_stress'])
    texts = {k: metadatareader.text_dict[k] for k in metadatareader.filenames}
    key_list = list(texts.keys())
    
    print('\nPHONEMIZING')
    batch_size = args.phonemizer_batch_size
    failed_files = []
    phonemized_data = {}
    for i in tqdm.tqdm(range(0, len(key_list) + batch_size, batch_size)):
        batch_keys = key_list[i:i + batch_size]
        try:
            batch_text = [texts[k] for k in batch_keys]
            if len(batch_text) == 0:
                break
            phonemized_batch = text_proc.phonemizer(batch_text, njobs=args.phonemizer_parallel_jobs)
            phonemized_data.update(dict(zip(batch_keys, phonemized_batch)))
        except:
            failed_files.extend(batch_keys)
    print(f'\nFailed to process {len(failed_files)} files. Retrying with reduced settings')
    print(f'{failed_files}')
    re_failed = []
    for file in tqdm.tqdm(failed_files):  # phonemizer sometimes breaks when computing batches or when multiproc.
        text = texts[file]
        try:
            phonemized_text = text_proc.phonemizer(text, njobs=1)
            phonemized_data.update({file: phonemized_text})
        except:
            re_failed.append(file)
            
    train_len -= len(re_failed)
    if len(re_failed)> 0:
        print(f'\nCould not phonemize {len(re_failed)} files. Excluding the following from training set.')
        print(re_failed)
    
    print('\nPhonemized metadata samples:')
    for i in sample_items:
        print(f'{i}:{phonemized_data[i]}')
    
    new_metadata = [''.join([key, '|', phonemized_data[key], '\n']) for key in phonemized_data]
    shuffled_metadata = np.random.permutation(new_metadata)
    train_metadata = shuffled_metadata[0:train_len]
    test_metadata = shuffled_metadata[-test_len:]
    
    with open(phonemized_metadata_path, 'w+', encoding='utf-8') as file:
        file.writelines(new_metadata)
    with open(train_metadata_path, 'w+', encoding='utf-8') as file:
        file.writelines(train_metadata)
    with open(test_metadata_path, 'w+', encoding='utf-8') as file:
        file.writelines(test_metadata)
    # some checks
    assert metadata_len - len(re_failed) == len(set(list(phonemized_data.keys()))), \
        f'Length of metadata ({metadata_len - len(re_failed)} ) does not match the length of the phoneme array ({len(set(list(phonemized_data.keys())))}). Check for empty text lines in metadata.'
    assert len(train_metadata) + len(test_metadata) == (metadata_len - len(re_failed)), \
        f'Train and/or validation lengths incorrect. ({len(train_metadata)} + {len(test_metadata)} != {metadata_len - len(re_failed)})'
print('\nDone')
