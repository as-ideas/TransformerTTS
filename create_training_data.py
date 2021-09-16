import argparse
from pathlib import Path
import pickle

import numpy as np
from p_tqdm import p_uimap, p_umap

from utils.logging_utils import SummaryManager
from data.text import TextToTokens
from data.datasets import DataReader
from utils.training_config_manager import TrainingConfigManager
from data.audio import Audio
from data.text.symbols import _alphabet, all_phonemes

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--skip_phonemization', action='store_true')
parser.add_argument('--skip_mels', action='store_true')

args = parser.parse_args()
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))

phonemized_flag = args.skip_phonemization  # If the input is already in the form of phonemes, this flag will be set (User wanting to skip phonemization indicates that the input is already phonemized)

cm = TrainingConfigManager(args.config, aligner=True)
cm.create_remove_dirs()
metadatareader = DataReader.from_config(cm, kind='original', scan_wavs=True)
summary_manager = SummaryManager(model=None, log_dir=cm.log_dir / 'data_preprocessing', config=cm.config,
                                 default_writer='data_preprocessing')
file_ids_from_wavs = list(metadatareader.wav_paths.keys())
print(f"Reading wavs from {metadatareader.wav_directory}")
print(f"Reading metadata from {metadatareader.metadata_path}")
print(f'\nFound {len(metadatareader.filenames)} metadata lines.')
print(f'\nFound {len(file_ids_from_wavs)} wav files.')
cross_file_ids = [fid for fid in file_ids_from_wavs if fid in metadatareader.filenames]
print(f'\nThere are {len(cross_file_ids)} wav file names that correspond to metadata lines.')

if not args.skip_mels:
    
    def process_wav(wav_path: Path):
        file_name = wav_path.stem
        y, sr = audio.load_wav(str(wav_path))
        pitch = audio.extract_pitch(y)
        mel = audio.mel_spectrogram(y)
        assert mel.shape[1] == audio.config['mel_channels'], len(mel.shape) == 2
        assert mel.shape[0] == pitch.shape[0], f'{mel.shape[0]} == {pitch.shape[0]} (wav {y.shape})'
        mel_path = (cm.mel_dir / file_name).with_suffix('.npy')
        pitch_path = (cm.pitch_dir / file_name).with_suffix('.npy')
        np.save(mel_path, mel)
        np.save(pitch_path, pitch)
        return {'fname': file_name, 'mel.len': mel.shape[0], 'pitch.path': pitch_path, 'pitch': pitch}
    
    
    print(f"\nMels will be stored stored under")
    print(f"{cm.mel_dir}")
    audio = Audio.from_config(config=cm.config)
    wav_files = [metadatareader.wav_paths[k] for k in cross_file_ids]
    len_dict = {}
    remove_files = []
    mel_lens = []
    pitches = {}
    wav_iter = p_uimap(process_wav, wav_files)
    for out_dict in wav_iter:
        len_dict.update({out_dict['fname']: out_dict['mel.len']})
        pitches.update({out_dict['pitch.path']: out_dict['pitch']})
        if out_dict['mel.len'] > cm.config['max_mel_len'] or out_dict['mel.len'] < cm.config['min_mel_len']:
            remove_files.append(out_dict['fname'])
        else:
            mel_lens.append(out_dict['mel.len'])
    
    
    def normalize_pitch_vectors(pitch_vecs):
        nonzeros = np.concatenate([v[np.where(v != 0.0)[0]]
                                   for v in pitch_vecs.values()])
        mean, std = np.mean(nonzeros), np.std(nonzeros)
        return mean, std
    
    
    def process_pitches(item: tuple):
        fname, pitch = item
        zero_idxs = np.where(pitch == 0.0)[0]
        pitch -= mean
        pitch /= std
        pitch[zero_idxs] = 0.0
        np.save(fname, pitch)
    
    
    mean, std = normalize_pitch_vectors(pitches)
    pickle.dump({'pitch_mean': mean, 'pitch_std': std}, open(cm.data_dir / 'pitch_stats.pkl', 'wb'))
    pitch_iter = p_umap(process_pitches, pitches.items())
    
    pickle.dump(len_dict, open(cm.data_dir / 'mel_len.pkl', 'wb'))
    pickle.dump(remove_files, open(cm.data_dir / 'under-over_sized_mels.pkl', 'wb'))
    summary_manager.add_histogram('Mel Lengths', values=np.array(mel_lens))
    total_mel_len = np.sum(mel_lens)
    total_wav_len = total_mel_len * audio.config['hop_length']
    summary_manager.display_scalar('Total duration (hours)',
                                   scalar_value=total_wav_len / audio.config['sampling_rate'] / 60. ** 2)
    
def get_short_files(phonemized=False):
    if not phonemized:
        symbol_list = _alphabet
    else:
        symbol_list = all_phonemes
        
    filter_metadata = []
    for fname in cross_file_ids:
        item = metadatareader.text_dict[fname]
        non_p = [c for c in item if c in symbol_list]
        if len(non_p) < 1:
            filter_metadata.append(fname)
    if len(filter_metadata) > 0:
        print(f'Removing {len(filter_metadata)} suspiciously short line(s):')
        for fname in filter_metadata:
            print(f'{fname}: {metadatareader.text_dict[fname]}')
    return filter_metadata

remove_files = pickle.load(open(cm.data_dir / 'under-over_sized_mels.pkl', 'rb'))
phonemized_metadata_path = cm.phonemized_metadata_path
train_metadata_path = cm.train_metadata_path
test_metadata_path = cm.valid_metadata_path
print(f'\nReading metadata from {metadatareader.metadata_path}')
print(f'\nFound {len(metadatareader.filenames)} lines.')

filter_metadata = get_short_files(phonemized=phonemized_flag)
remove_files += filter_metadata
print(f'\nRemoving {len(remove_files)} line(s) due to mel filtering.')
metadata_file_ids = [fname for fname in cross_file_ids if fname not in remove_files]
metadata_len = len(metadata_file_ids)
sample_items = np.random.choice(metadata_file_ids, 5)
test_len = cm.config['n_test']
train_len = metadata_len - test_len
print(f'\nMetadata contains {metadata_len} lines.')
print(f'\nFiles will be stored under {cm.data_dir}')
print(f' - all: {phonemized_metadata_path}')
print(f' - {train_len} training lines: {train_metadata_path}')
print(f' - {test_len} validation lines: {test_metadata_path}')

# run cleaner on raw text
text_proc = TextToTokens.default(cm.config['phoneme_language'], add_start_end=False,
                                    with_stress=cm.config['with_stress'], model_breathing=cm.config['model_breathing'],
                                    njobs=1)


def process_phonemes(file_id):
    text = metadatareader.text_dict[file_id]
    try:
        phon = text_proc.phonemizer(text, only_preprocess=phonemized_flag)
    except Exception as e:
        print(f'{e}\nFile id {file_id}')
        raise BrokenPipeError
    return (file_id, phon)


print('\nPHONEMIZING')
phonemized_data = {}
phon_iter = p_uimap(process_phonemes, metadata_file_ids)
for (file_id, phonemes) in phon_iter:
    phonemized_data.update({file_id: phonemes})

print('\nPhonemized metadata samples:')
for i in sample_items:
    print(f'{i}:{phonemized_data[i]}')
    summary_manager.add_text(f'{i}/phonemes', text=phonemized_data[i])

new_metadata = [f'{k}|{v}\n' for k, v in phonemized_data.items()]
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
    
print('\n Done')
