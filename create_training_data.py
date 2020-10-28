import argparse
from pathlib import Path
import pickle

import numpy as np
from p_tqdm import p_uimap

from utils.logging_utils import SummaryManager
from preprocessing.text import TextToTokens
from preprocessing.datasets import DataReader
from utils.config_manager import Config
from utils.audio import Audio
from preprocessing.text.symbols import _alphabet

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--skip_phonemes', action='store_true')
parser.add_argument('--skip_mels', action='store_true')

args = parser.parse_args()
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))

cm = Config(args.config, model_kind='autoregressive')
cm.create_remove_dirs()
metadatareader = DataReader.from_config(cm, kind='original', scan_wavs=True)
summary_manager = SummaryManager(model=None, log_dir=cm.log_dir / 'data_preprocessing', config=cm.config,
                                 default_writer='data_preprocessing')
if not args.skip_mels:
    
    def process_wav(wav_path: Path):
        file_name = wav_path.stem
        y, sr = audio.load_wav(str(wav_path))
        mel = audio.mel_spectrogram(y)
        assert mel.shape[1] == audio.config['mel_channels'], len(mel.shape) == 2
        mel_path = (cm.mel_dir / file_name).with_suffix('.npy')
        np.save(mel_path, mel)
        return (file_name, mel.shape[0])
    
    
    print(f"Creating mels from all wavs found in {metadatareader.data_directory}")
    print(f"\nMels will be stored stored under")
    print(f"{cm.mel_dir}")
    (cm.mel_dir).mkdir(exist_ok=True)
    audio = Audio(config=cm.config)
    wav_files = [metadatareader.wav_paths[k] for k in metadatareader.wav_paths]
    len_dict = {}
    remove_files = []
    mel_lens = []
    wav_iter = p_uimap(process_wav, wav_files)
    for (fname, mel_len) in wav_iter:
        len_dict.update({fname: mel_len})
        if mel_len > cm.config['max_mel_len'] or mel_len < cm.config['min_mel_len']:
            remove_files.append(fname)
        else:
            mel_lens.append(mel_len)
    
    pickle.dump(len_dict, open(cm.data_dir / 'mel_len.pkl', 'wb'))
    pickle.dump(remove_files, open(cm.data_dir / 'under-over_sized_mels.pkl', 'wb'))
    summary_manager.add_histogram('Mel Lengths', values=np.array(mel_lens))
    total_mel_len = np.sum(mel_lens)
    total_wav_len = total_mel_len * audio.config['hop_length']
    summary_manager.display_scalar('Total duration (hours)',
                                   scalar_value=total_wav_len / audio.config['sampling_rate'] / 60. ** 2)

if not args.skip_phonemes:
    remove_files = pickle.load(open(cm.data_dir / 'under-over_sized_mels.pkl', 'rb'))
    phonemized_metadata_path = Path(cm.data_dir) / 'phonemized_metadata.txt'
    train_metadata_path = Path(cm.data_dir) / cm.config['train_metadata_filename']
    test_metadata_path = Path(cm.data_dir) / cm.config['valid_metadata_filename']
    print(f'\nReading metadata from {metadatareader.metadata_path}')
    print(f'\nFound {len(metadatareader.filenames)} lines.')
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
        summary_manager.add_text(f'Metadata samples/{i}', text=metadatareader.text_dict[i])
    
    # run cleaner on raw text
    text_proc = TextToTokens.default(cm.config['phoneme_language'], add_start_end=False,
                                     with_stress=cm.config['with_stress'], njobs=1)
    
    
    def process_phonemes(file_id):
        text = metadatareader.text_dict[file_id]
        try:
            phon = text_proc.phonemizer(text)
        except Exception as e:
            print(f'{e}\nFile id {file_id}')
            raise BrokenPipeError
        return (file_id, phon)
    
    
    print('\nPHONEMIZING')
    phonemized_data = {}
    phon_iter = p_uimap(process_phonemes, metadatareader.filenames)
    for (file_id, phonemes) in phon_iter:
        phonemized_data.update({file_id: phonemes})
    
    print('\nPhonemized metadata samples:')
    for i in sample_items:
        print(f'{i}:{phonemized_data[i]}')
        summary_manager.add_text(f'Phonemized samples/{i}', text=phonemized_data[i])
    
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
    assert metadata_len == len(set(list(phonemized_data.keys()))), \
        f'Length of metadata ({metadata_len}) does not match the length of the phoneme array ({len(set(list(phonemized_data.keys())))}). Check for empty text lines in metadata.'
    assert len(train_metadata) + len(test_metadata) == metadata_len, \
        f'Train and/or validation lengths incorrect. ({len(train_metadata)} + {len(test_metadata)} != {metadata_len})'
print('\nDone')
