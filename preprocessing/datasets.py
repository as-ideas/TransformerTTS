from pathlib import Path
from random import Random
from typing import List, Union

import numpy as np
import tensorflow as tf

from utils.config_manager import Config
from preprocessing.text.tokenizer import Tokenizer
from preprocessing.metadata_readers import get_preprocessor_by_name


def get_files(path: Union[Path, str], extension='.wav') -> List[Path]:
    """ Get all files from all subdirs with given extension. """
    path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))


class DataReader:
    """
    Reads dataset folder and constructs three useful objects:
        text_dict: {filename: text}
        wav_paths: {filename: path/to/filename.wav}
        filenames: [filename1, filename2, ...]
        
    IMPORTANT: Use only for information available from source dataset, not for
    training data.
    """
    
    def __init__(self, data_directory: str, metadata_path: str, metadata_reading_function=None, scan_wavs=False):
        self.metadata_reading_function = metadata_reading_function
        self.data_directory = Path(data_directory)
        self.metadata_path = Path(metadata_path)
        self.text_dict = self.metadata_reading_function(self.metadata_path)
        self.filenames = list(self.text_dict.keys())
        if scan_wavs:
            all_wavs = get_files(self.data_directory, extension='.wav')
            self.wav_paths = {w.with_suffix('').name: w for w in all_wavs}
    
    @classmethod
    def from_config(cls, config_manager: Config, kind: str, scan_wavs=False):
        kinds = ['original', 'phonemized', 'train', 'valid']
        if kind not in kinds:
            raise ValueError(f'Invalid kind type. Expected one of: {kinds}')
        
        reader = get_preprocessor_by_name(config_manager.config['data_name'])
        if kind == 'train':
            metadata = config_manager.train_metadata_path
        elif kind == 'original':
            metadata = config_manager.metadata_path
        elif kind == 'valid':
            metadata = config_manager.valid_metadata_path
        elif kind == 'phonemized':
            metadata = config_manager.phonemized_metadata_path
        
        return cls(data_directory=config_manager.dataset_dir,
                   metadata_reading_function=reader,
                   metadata_path=metadata,
                   scan_wavs=scan_wavs)


class TextMelDataset:
    def __init__(self,
                 data_reader: DataReader,
                 preprocessor,
                 mel_directory: str):
        self.metadata_reader = data_reader
        self.preprocessor = preprocessor
        self.mel_directory = Path(mel_directory)
    
    def _read_sample(self, sample_name):
        text = self.metadata_reader.text_dict[sample_name]
        mel = np.load((self.mel_directory / sample_name).with_suffix('.npy').as_posix())
        return mel, text
    
    def _process_sample(self, sample_name):
        mel, text = self._read_sample(sample_name)
        return self.preprocessor(mel=mel, text=text, sample_name=sample_name)
    
    def get_dataset(self, bucket_batch_sizes, bucket_boundaries, shuffle=True, drop_remainder=False):
        return Dataset(
            samples=self.metadata_reader.filenames,
            preprocessor=self._process_sample,
            output_types=self.preprocessor.output_types,
            padded_shapes=self.preprocessor.padded_shapes,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
            len_function=self.preprocessor.get_sample_length,
            bucket_batch_sizes=bucket_batch_sizes,
            bucket_boundaries=bucket_boundaries)
    
    @classmethod
    def from_config(cls,
                    config: Config,
                    preprocessor,
                    kind: str,
                    mel_directory: str = None, ):
        kinds = ['original', 'phonemized', 'train', 'valid']
        if kind not in kinds:
            raise ValueError(f'Invalid kind type. Expected one of: {kinds}')
        if mel_directory is None:
            mel_directory = config.mel_dir
        metadata_reader = DataReader.from_config(config, kind=kind)
        return cls(preprocessor=preprocessor,
                   data_reader=metadata_reader,
                   mel_directory=mel_directory)


class TextMelDurDataset:
    def __init__(self,
                 data_reader: DataReader,
                 preprocessor,
                 mel_directory: str,
                 duration_directory: str):
        self.metadata_reader = data_reader
        self.preprocessor = preprocessor
        self.mel_directory = Path(mel_directory)
        self.duration_directory = Path(duration_directory)
    
    def _read_sample(self, sample_name: str):
        text = self.metadata_reader.text_dict[sample_name]
        mel = np.load((self.mel_directory / sample_name).with_suffix('.npy').as_posix())
        durations = np.load(
            (self.duration_directory / sample_name).with_suffix('.npy').as_posix())
        return mel, text, durations
    
    def _process_sample(self, sample_name: str):
        mel, text, durations = self._read_sample(sample_name)
        return self.preprocessor(mel=mel, text=text, durations=durations, sample_name=sample_name)
    
    def get_dataset(self, bucket_batch_sizes, bucket_boundaries, shuffle=True, drop_remainder=False):
        return Dataset(
            samples=self.metadata_reader.filenames,
            preprocessor=self._process_sample,
            output_types=self.preprocessor.output_types,
            padded_shapes=self.preprocessor.padded_shapes,
            len_function=self.preprocessor.get_sample_length,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
            bucket_batch_sizes=bucket_batch_sizes,
            bucket_boundaries=bucket_boundaries)
    
    @classmethod
    def from_config(cls,
                    config: Config,
                    preprocessor,
                    kind: str,
                    mel_directory: str = None,
                    duration_directory: str = None):
        kinds = ['phonemized', 'train', 'valid']
        if kind not in kinds:
            raise ValueError(f'Invalid kind type. Expected one of: {kinds}')
        if mel_directory is None:
            mel_directory = config.mel_dir
        if duration_directory is None:
            duration_directory = config.data_dir / 'durations'
        metadata_reader = DataReader.from_config(config,
                                                 kind=kind)
        return cls(preprocessor=preprocessor,
                   data_reader=metadata_reader,
                   mel_directory=mel_directory,
                   duration_directory=duration_directory)


class Dataset:
    """ Model digestible dataset. """
    
    def __init__(self,
                 samples: list,
                 preprocessor,
                 len_function,
                 padded_shapes: tuple,
                 output_types: tuple,
                 bucket_boundaries: list,
                 bucket_batch_sizes: list,
                 padding_values: tuple = None,
                 shuffle=True,
                 drop_remainder=True,
                 seed=42):
        self._random = Random(seed)
        self._samples = samples[:]
        self.preprocessor = preprocessor
        dataset = tf.data.Dataset.from_generator(lambda: self._datagen(shuffle),
                                                 output_types=output_types)
        # TODO: pass bin args
        binned_data = dataset.apply(
            tf.data.experimental.bucket_by_sequence_length(
                len_function,
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                padded_shapes=padded_shapes,
                drop_remainder=drop_remainder,
                padding_values=padding_values
            ))
        self.dataset = binned_data
        self.data_iter = iter(binned_data.repeat(-1))
    
    def next_batch(self):
        return next(self.data_iter)
    
    def all_batches(self):
        return iter(self.dataset)
    
    def _datagen(self, shuffle):
        """
        Shuffle once before generating to avoid buffering
        """
        samples = self._samples[:]
        if shuffle:
            self._random.shuffle(samples)
        return (self.preprocessor(s) for s in samples)


class ForwardPreprocessor:
    def __init__(self, mel_channels, tokenizer: Tokenizer):
        self.output_types = (tf.float32, tf.int32, tf.int32, tf.string)
        self.padded_shapes = ([None, mel_channels], [None], [None], [])
        self.tokenizer = tokenizer
    
    def __call__(self, text, mel, durations, sample_name):
        encoded_phonemes = self.tokenizer(text)
        return mel, encoded_phonemes, durations, sample_name
    
    def get_sample_length(self, mel, encoded_phonemes, durations, sample_name):
        return tf.shape(mel)[0]
    
    @classmethod
    def from_config(cls, config: Config, tokenizer: Tokenizer):
        return cls(mel_channels=config.config['mel_channels'],
                   tokenizer=tokenizer)


class AutoregressivePreprocessor:
    
    def __init__(self,
                 mel_channels: int,
                 mel_start_value: float,
                 mel_end_value: float,
                 tokenizer: Tokenizer):
        self.output_types = (tf.float32, tf.int32, tf.int32, tf.string)
        self.padded_shapes = ([None, mel_channels], [None], [None], [])
        self.start_vec = np.ones((1, mel_channels)) * mel_start_value
        self.end_vec = np.ones((1, mel_channels)) * mel_end_value
        self.tokenizer = tokenizer
    
    def __call__(self, mel, text, sample_name):
        encoded_phonemes = self.tokenizer(text)
        norm_mel = np.concatenate([self.start_vec, mel, self.end_vec], axis=0)
        stop_probs = np.ones((norm_mel.shape[0]))
        stop_probs[-1] = 2
        return norm_mel, encoded_phonemes, stop_probs, sample_name
    
    def get_sample_length(self, norm_mel, encoded_phonemes, stop_probs, sample_name):
        return tf.shape(norm_mel)[0]
    
    @classmethod
    def from_config(cls, config: Config, tokenizer: Tokenizer):
        return cls(mel_channels=config.config['mel_channels'],
                   mel_start_value=config.config['mel_start_value'],
                   mel_end_value=config.config['mel_end_value'],
                   tokenizer=tokenizer)


if __name__ == '__main__':
    ljspeech_folder = '/Volumes/data/datasets/LJSpeech-1.1'
    # metadata_path = '/Volumes/data/datasets/LJSpeech-1.1/metadata.csv'
    metadata_path = '/Volumes/data/datasets/LJSpeech-1.1/transformer_tts/phonemized_metadata.txt'
    metadata_reader = get_preprocessor_by_name('ljspeech')
    data_reader = DataReader(data_directory=ljspeech_folder, metadata_path=metadata_path,
                             metadata_reading_function=metadata_reader, scan_wavs=True)
    key_list = data_reader.filenames
    print('metadata head')
    for key in key_list[:5]:
        print(f'{key}: {data_reader.text_dict[key]}')
    print('metadata tail')
    for key in key_list[-5:]:
        print(f'{key}: {data_reader.text_dict[key]}')
    print('wav paths head')
    for key in key_list[:5]:
        print(f'{key}: {data_reader.wav_paths[key]}')
    print('wav paths tail')
    for key in key_list[-5:]:
        print(f'{key}: {data_reader.wav_paths[key]}')
    mel_dir = Path('/Volumes/data/datasets/LJSpeech-1.1/transformer_tts/mels')
    if mel_dir.exists():
        from preprocessing.text.tokenizer import Tokenizer
        from preprocessing.text.symbols import all_phonemes
        
        tokenizer = Tokenizer()
        preprocessor = AutoregressivePreprocessor(mel_channels=80,
                                                  mel_start_value=.5,
                                                  mel_end_value=-.5,
                                                  tokenizer=tokenizer)
        dataset_creator = TextMelDataset(data_reader=data_reader,
                                         preprocessor=preprocessor,
                                         mel_directory=mel_dir)
        dataset = dataset_creator.get_dataset(shuffle=True, drop_remainder=False, bucket_boundaries=[500], bucket_batch_sizes=[6,6])
        for i in range(10):
            batch = dataset.next_batch()
            bsh = tf.shape(batch[0])
            print(f'bs{bsh[0]} | len {bsh[1]}')
