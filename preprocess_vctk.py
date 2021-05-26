import argparse
import pickle

import tensorflow_datasets as tfds
from resemblyzer import VoiceEncoder
import numpy as np
from p_tqdm import p_uimap, p_umap
from tqdm import tqdm

from utils.logging_utils import SummaryManager
from utils.training_config_manager import TrainingConfigManager
from data.audio import Audio

np.random.seed(42)


def filter_speaker(x):
    return x['speaker'] == 1


def filter_non_speaker(x):
    return x['speaker'] != 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--skip_phonemes', action='store_true')
    parser.add_argument('--skip_mels', action='store_true')
    
    args = parser.parse_args()
    
    dl_config = tfds.download.DownloadConfig(verify_ssl=False)
    dataset = tfds.load('vctk', data_dir='/data/datasets/',
                        download_and_prepare_kwargs={'download_config': dl_config, })['train']#, split='train[99%:]')
    cm = TrainingConfigManager('config/training_config_remote.yaml', aligner=True)
    cm.create_remove_dirs()
    summary_manager = SummaryManager(model=None, log_dir=cm.log_dir / 'data_preprocessing', config=cm.config,
                                     default_writer='data_preprocessing')
    
    if not args.skip_mels:
        audio = Audio.from_config(config=cm.config)
        encoder = VoiceEncoder()
        
        len_dict = {}
        remove_files = []
        mel_lens = []
        pitches = {}
        
        
        def preprocess_data(x):
            speech = x['speech'].numpy().astype(float)
            file_name = x['id'].numpy().decode("utf-8")
            y = audio.preprocess(speech)
            y = np.array(y)
            pitch = audio.extract_pitch(y)
            mel = audio.mel_spectrogram(y)
            mel_path = (cm.mel_dir / file_name).with_suffix('.npy')
            pitch_path = (cm.pitch_dir / file_name).with_suffix('.npy')
            np.save(mel_path, mel)
            np.save(pitch_path, pitch)
            return {'fname': file_name, 'mel.len': mel.shape[0], 'pitch.path': pitch_path, 'pitch': pitch}
        
        
        wav_iter = p_uimap(preprocess_data, dataset)
        print('Storing mel spectrograms and pitch values')
        for out_dict in wav_iter:
            len_dict.update({out_dict['fname']: out_dict['mel.len']})
            pitches.update({out_dict['pitch.path']: out_dict['pitch']})
            # if out_dict['mel.len'] > cm.config['max_mel_len'] or out_dict['mel.len'] < cm.config['min_mel_len']:
            #     remove_files.append(out_dict['fname'])
            # else:
            #     mel_lens.append(out_dict['mel.len'])
            mel_lens.append(out_dict['mel.len'])
        
        
        def save_embeddings(x):
            speech = x['speech'].numpy().astype(float)
            file_name = x['id'].numpy().decode("utf-8")
            y = audio.preprocess(speech)
            y = np.array(y)
            embed = encoder.embed_utterance(y)
            embed_path = (cm.embed_dir / file_name).with_suffix('.npy')
            np.save(embed_path, embed)
        
        
        print('Storing embeddings')
        iterator = tqdm(dataset)
        for item in iterator:
            save_embeddings(item)
        
        
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
        # pickle.dump(remove_files, open(cm.data_dir / 'under-over_sized_mels.pkl', 'wb'))
    
    if not args.skip_phonemes:
        # remove_files = pickle.load(open(cm.data_dir / 'under-over_sized_mels.pkl', 'rb'))
        phonemized_metadata_path = cm.phonemized_metadata_path
        train_metadata_path = cm.train_metadata_path
        test_metadata_path = cm.valid_metadata_path
        print("splitting data")
        train_set = tfds.load('vctk', data_dir='/data/datasets/',
                              download_and_prepare_kwargs={'download_config': dl_config, }, split='train[:98%]')
        validation_set = tfds.load('vctk', data_dir='/data/datasets/',
                                   download_and_prepare_kwargs={'download_config': dl_config, }, split='train[98%:]')
        
        
        print('filtering data')
        final_training_set = train_set.filter(filter_non_speaker)
        additional_validation_set = train_set.filter(filter_speaker)
        validation_set = validation_set.concatenate(additional_validation_set)
        print('building metadata')
        training_ids = [item['id'].numpy().decode('utf-8') for item in final_training_set]
        validation_ids = [item['id'].numpy().decode('utf-8') for item in validation_set]
        all_metadata = {item['id'].numpy().decode('utf-8'): item['text'].numpy().decode('utf-8') for item in dataset}
        
        sample_items = np.random.choice(list(all_metadata.keys()), 5)
        print(f'\nFiles will be stored under {cm.data_dir}')
        print(f' - all: {phonemized_metadata_path} ({len(all_metadata)}')
        print(f' - {len(training_ids)} training lines: {train_metadata_path}')
        print(f' - {len(validation_ids)} validation lines: {test_metadata_path}')
        
        print('\nMetadata samples:')
        for i in sample_items:
            print(f'{i}:{all_metadata[i]}')
            summary_manager.add_text(f'{i}/text', text=all_metadata[i])
        from data.text import TextToTokens
        
        # run cleaner on raw text
        text_proc = TextToTokens.default(cm.config['phoneme_language'], add_start_end=False,
                                         with_stress=cm.config['with_stress'],
                                         model_breathing=cm.config['model_breathing'],
                                         njobs=1)
        
        
        def process_phonemes(file_id):
            text = all_metadata[file_id]
            try:
                phon = text_proc.phonemizer(text)
            except Exception as e:
                print(f'{e}\nFile id {file_id}')
                raise BrokenPipeError
            return (file_id, phon)
        
        
        print('\nPHONEMIZING')
        phonemized_data = {}
        phon_iter = p_uimap(process_phonemes, all_metadata.keys())
        for (file_id, phonemes) in phon_iter:
            phonemized_data.update({file_id: phonemes})
        
        print('\nPhonemized metadata samples:')
        for i in sample_items:
            print(f'{i}:{phonemized_data[i]}')
            summary_manager.add_text(f'{i}/phonemes', text=phonemized_data[i])
        
        new_metadata = [f'{k}|{v}\n' for k, v in phonemized_data.items()]
        with open(phonemized_metadata_path, 'w+', encoding='utf-8') as file:
            file.writelines(new_metadata)
            
        train_metadata = [f'{k}|{phonemized_data[k]}\n' for k in training_ids]
        test_metadata = [f'{k}|{phonemized_data[k]}\n' for k in validation_ids]

        
        with open(train_metadata_path, 'w+', encoding='utf-8') as file:
            file.writelines(train_metadata)
        with open(test_metadata_path, 'w+', encoding='utf-8') as file:
            file.writelines(test_metadata)
        # some checks
        assert len(all_metadata) == len(set(list(phonemized_data.keys()))), \
            f'Length of metadata ({len(all_metadata)}) does not match the length of the phoneme array ({len(set(list(phonemized_data.keys())))}). Check for empty text lines in metadata.'
        assert len(train_metadata) + len(test_metadata) == len(all_metadata), \
            f'Train and/or validation lengths incorrect. ({len(train_metadata)} + {len(test_metadata)} != {len(all_metadata)})'
