from pathlib import Path
from random import Random

import numpy as np
import tensorflow as tf

from utils.config_manager import Config
from utils.audio import Audio
from preprocessing.text.tokenizer import Tokenizer


class MetadataReader:
    def __init__(self,
                 data_directory: str,
                 metadata_path: str,
                 wav_path: str,
                 metadata_reading_function=None, ):
        if metadata_reading_function is not None:
            self.metadata_reading_function = metadata_reading_function
        else:
            self.metadata_reading_function = self._default_metadata_reader
        
        self.data_directory = Path(data_directory)
        self.metadata_path = Path(metadata_path)
        self.wav_directory = Path(wav_path)
        self.data = self._build_file_list()
    
    def _build_file_list(self):
        wav_list, text_list = self.metadata_reading_function(self.metadata_path)
        file_list = [x.with_suffix('').name for x in self.wav_directory.iterdir() if x.suffix == '.wav']
        for metadata_item in wav_list:
            assert metadata_item in file_list, f'Missing file: metadata item {metadata_item}, was not found in {self.wav_directory}.'
        return list(zip(wav_list, text_list))
    
    def _default_metadata_reader(self, metadata_path, column_sep='|'):
        wav_list = []
        text_list = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for l in f.readlines():
                l_split = l.split(column_sep)
                filename, text = l_split[0], l_split[1]
                if filename.endswith('.wav'):
                    filename = filename.split('.')[0]
                wav_list.append(filename)
                text_list.append(text)
        return wav_list, text_list
    
    @classmethod
    def default_original_from_config(cls, config_manager: Config, metadata_reading_function=None):
        return cls(data_directory=config_manager.train_datadir,
                   metadata_reading_function=metadata_reading_function,
                   metadata_path=config_manager.metadata_path,
                   wav_path=config_manager.wav_dir)
    
    @classmethod
    def default_all_phonemized_from_config(cls, config_manager: Config, metadata_reading_function=None):
        return cls(data_directory=config_manager.train_datadir,
                   metadata_reading_function=metadata_reading_function,
                   metadata_path=config_manager.phonemized_metadata_path,
                   wav_path=config_manager.wav_dir)
    
    @classmethod
    def default_training_from_config(cls, config_manager: Config, metadata_reading_function=None):
        return cls(data_directory=config_manager.train_datadir,
                   metadata_reading_function=metadata_reading_function,
                   metadata_path=config_manager.train_metadata_path,
                   wav_path=config_manager.wav_dir)
    
    @classmethod
    def default_validation_from_config(cls, config_manager: Config, metadata_reading_function=None):
        return cls(data_directory=config_manager.train_datadir,
                   metadata_reading_function=metadata_reading_function,
                   metadata_path=config_manager.valid_metadata_path,
                   wav_path=config_manager.wav_dir)


class TextMelDataset:
    def __init__(self,
                 metadata_reader: MetadataReader,
                 preprocessor,
                 mel_directory: str):
        self.metadata_reader = metadata_reader
        self.preprocessor = preprocessor
        self.mel_directory = Path(mel_directory)
    
    def _read_sample(self, sample):
        sample_name = sample[0]
        text = sample[1]
        mel = np.load((self.mel_directory / sample_name).with_suffix('.npy').as_posix())
        return mel, text, sample_name
    
    def _process_sample(self, sample):
        mel, text, sample_name = self._read_sample(sample)
        return self.preprocessor(mel=mel, text=text, sample_name=sample_name)
    
    def get_dataset(self, batch_size, shuffle=True, drop_remainder=False):
        return Dataset(
            samples=self.metadata_reader.data,
            preprocessor=self._process_sample,
            batch_size=batch_size,
            output_types=self.preprocessor.output_types,
            padded_shapes=self.preprocessor.padded_shapes,
            shuffle=shuffle,
            drop_remainder=drop_remainder)
    
    @classmethod
    def default_all_from_config(cls,
                                config: Config,
                                preprocessor,
                                mel_directory: str = None,
                                metadata_reading_function=None):
        if mel_directory is None:
            mel_directory = config.train_datadir / 'mels'
        metadata_reader = MetadataReader.default_all_phonemized_from_config(config,
                                                                            metadata_reading_function=metadata_reading_function)
        return cls(preprocessor=preprocessor,
                   metadata_reader=metadata_reader,
                   mel_directory=mel_directory)
    
    @classmethod
    def default_training_from_config(cls,
                                     config: Config,
                                     preprocessor,
                                     mel_directory: str = None,
                                     metadata_reading_function=None):
        if mel_directory is None:
            mel_directory = config.train_datadir / 'mels'
        metadata_reader = MetadataReader.default_training_from_config(config,
                                                                      metadata_reading_function=metadata_reading_function)
        return cls(preprocessor=preprocessor,
                   metadata_reader=metadata_reader,
                   mel_directory=mel_directory)
    
    @classmethod
    def default_validation_from_config(cls,
                                       config: Config,
                                       preprocessor,
                                       mel_directory: str = None,
                                       metadata_reading_function=None):
        if mel_directory is None:
            mel_directory = config.train_datadir / 'mels'
        metadata_reader = MetadataReader.default_validation_from_config(config,
                                                                        metadata_reading_function=metadata_reading_function)
        return cls(preprocessor=preprocessor,
                   metadata_reader=metadata_reader,
                   mel_directory=mel_directory)


class TextMelDurDataset:
    def __init__(self,
                 metadata_reader: MetadataReader,
                 preprocessor,
                 mel_directory: str):
        self.metadata_reader = metadata_reader
        self.preprocessor = preprocessor
        self.mel_directory = Path(mel_directory)
    
    def _read_sample(self, sample):
        sample_name = sample[0]
        text = sample[1]
        mel = np.load((self.mel_directory / sample_name).with_suffix('.npy').as_posix())
        durations = np.load(
            (self.metadata_reader.data_directory / 'durations' / sample_name).with_suffix('.npy').as_posix())
        return mel, text, durations, sample_name
    
    def _process_sample(self, sample):
        mel, text, durations, sample_name = self._read_sample(sample)
        return self.preprocessor(mel=mel, text=text, durations=durations, sample_name=sample_name)
    
    def get_dataset(self, batch_size, shuffle=True, drop_remainder=False):
        return Dataset(
            samples=self.metadata_reader.data,
            preprocessor=self._process_sample,
            batch_size=batch_size,
            output_types=self.preprocessor.output_types,
            padded_shapes=self.preprocessor.padded_shapes,
            shuffle=shuffle,
            drop_remainder=drop_remainder)
    
    @classmethod
    def default_all_from_config(cls,
                                config: Config,
                                preprocessor,
                                mel_directory: str = None,
                                metadata_reading_function=None):
        if mel_directory is None:
            mel_directory = config.train_datadir / 'mels'
        metadata_reader = MetadataReader.default_all_phonemized_from_config(config,
                                                                            metadata_reading_function=metadata_reading_function)
        return cls(preprocessor=preprocessor,
                   metadata_reader=metadata_reader,
                   mel_directory=mel_directory)
    
    @classmethod
    def default_training_from_config(cls,
                                     config: Config,
                                     preprocessor,
                                     mel_directory: str = None,
                                     metadata_reading_function=None):
        if mel_directory is None:
            mel_directory = config.train_datadir / 'mels'
        metadata_reader = MetadataReader.default_training_from_config(config,
                                                                      metadata_reading_function=metadata_reading_function)
        return cls(preprocessor=preprocessor,
                   metadata_reader=metadata_reader,
                   mel_directory=mel_directory)
    
    @classmethod
    def default_validation_from_config(cls,
                                       config: Config,
                                       preprocessor,
                                       mel_directory: str = None,
                                       metadata_reading_function=None):
        if mel_directory is None:
            mel_directory = config.train_datadir / 'mels'
        metadata_reader = MetadataReader.default_validation_from_config(config,
                                                                        metadata_reading_function=metadata_reading_function)
        return cls(preprocessor=preprocessor,
                   metadata_reader=metadata_reader,
                   mel_directory=mel_directory)


class MelWavDataset:
    def __init__(self,
                 metadata_reader: MetadataReader,
                 preprocessor,
                 audio_module: Audio,
                 mel_directory: str,
                 max_wav_len=None, ):
        
        self.audio = audio_module
        self.metadata_reader = metadata_reader
        self.preprocessor = preprocessor
        self.max_wav_len = max_wav_len
        self.hop_len = self.audio.config['hop_length']
        self.mel_directory = Path(mel_directory)
    
    def _read_sample(self, sample):
        sample_name = sample[0]
        y = np.load(
            (self.metadata_reader.data_directory / 'resampled_wavs' / sample_name).with_suffix('.npy').as_posix())
        mel = np.load((self.mel_directory / sample_name).with_suffix('.npy').as_posix())
        if self.max_wav_len is not None:
            y_len = tf.shape(y)[0]
            offset = tf.random.uniform([1], 0, max(1, y_len - self.max_wav_len), dtype=tf.int32)[0]
            offset = (offset // self.hop_len) * self.hop_len  # ensure wav mel correspondence
            y = y[offset: offset + self.max_wav_len]
            mel = mel[offset // self.hop_len:(offset + self.max_wav_len) // self.hop_len]
        return mel, y, sample_name
    
    def _process_sample(self, sample):
        mel, wav, sample_name = self._read_sample(sample)
        return self.preprocessor(mel=mel, wav=wav, sample_name=sample_name)
    
    def get_dataset(self, batch_size, shuffle=True, drop_remainder=False):
        return Dataset(
            samples=self.metadata_reader.data,
            preprocessor=self._process_sample,
            batch_size=batch_size,
            output_types=self.preprocessor.output_types,
            padded_shapes=self.preprocessor.padded_shapes,
            shuffle=shuffle,
            drop_remainder=drop_remainder)
    
    @classmethod
    def default_all_from_config(cls,
                                config: Config,
                                preprocessor,
                                mel_directory: str,
                                metadata_reading_function=None,
                                max_wav_len: int = None):
        if mel_directory is None:
            if config.config['use_GT'] is True:
                mel_directory = config.train_datadir / 'mels'
            else:
                mel_directory = config.train_datadir / 'forward_mels'
        audio = Audio(config.config)
        if max_wav_len is None:
            max_wav_len = config.config['max_wav_segment_lenght']
        elif max_wav_len == -1:
            max_wav_len = None
        metadata_reader = MetadataReader.default_all_phonemized_from_config(config,
                                                                            metadata_reading_function=metadata_reading_function)
        return cls(preprocessor=preprocessor,
                   audio_module=audio,
                   metadata_reader=metadata_reader,
                   max_wav_len=max_wav_len,
                   mel_directory=mel_directory)
    
    @classmethod
    def default_training_from_config(cls,
                                     config: Config,
                                     preprocessor,
                                     metadata_reading_function=None,
                                     max_wav_len: int = None,
                                     mel_directory: str = None):
        if mel_directory is None:
            if config.config['use_GT'] is True:
                mel_directory = config.train_datadir / 'mels'
            else:
                mel_directory = config.train_datadir / 'forward_mels'
        audio = Audio(config.config)
        if max_wav_len is None:
            max_wav_len = config.config['max_wav_segment_lenght']
        elif max_wav_len == -1:
            max_wav_len = None
        metadata_reader = MetadataReader.default_training_from_config(config,
                                                                      metadata_reading_function=metadata_reading_function)
        return cls(preprocessor=preprocessor,
                   audio_module=audio,
                   metadata_reader=metadata_reader,
                   max_wav_len=max_wav_len,
                   mel_directory=mel_directory)
    
    @classmethod
    def default_validation_from_config(cls,
                                       config: Config,
                                       preprocessor,
                                       metadata_reading_function=None,
                                       max_wav_len: int = None,
                                       mel_directory: str = None):
        if mel_directory is None:
            if config.config['use_GT'] is True:
                mel_directory = config.train_datadir / 'mels'
            else:
                mel_directory = config.train_datadir / 'forward_mels'
        
        audio = Audio(config.config)
        if max_wav_len is None:
            max_wav_len = config.config['max_wav_segment_lenght']
        elif max_wav_len == -1:
            max_wav_len = None
        metadata_reader = MetadataReader.default_validation_from_config(config,
                                                                        metadata_reading_function=metadata_reading_function)
        return cls(preprocessor=preprocessor,
                   audio_module=audio,
                   metadata_reader=metadata_reader,
                   max_wav_len=max_wav_len,
                   mel_directory=mel_directory)


class Dataset:
    """ Model digestible dataset. """
    
    def __init__(self,
                 samples: list,
                 preprocessor,
                 batch_size: int,
                 padded_shapes: tuple,
                 output_types: tuple,
                 padding_values: tuple = None,
                 shuffle=True,
                 drop_remainder=True,
                 seed=42):
        self._random = Random(seed)
        self._samples = samples[:]
        self.preprocessor = preprocessor
        dataset = tf.data.Dataset.from_generator(lambda: self._datagen(shuffle),
                                                 output_types=output_types)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=padded_shapes,
                                       drop_remainder=drop_remainder,
                                       padding_values=padding_values)
        self.dataset = dataset
        self.data_iter = iter(dataset.repeat(-1))
    
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


class MelGANPreprocessor:
    def __init__(self, config):
        self.output_types = (tf.float32, tf.float32, tf.string)
        self.padded_shapes = ([-1, config['mel_channels']], [-1, 1], [])
        self.padding_values = (
            tf.constant(0, dtype=tf.float32), tf.constant(-1, dtype=tf.float32))  # TODO: get padding from config
    
    def __call__(self, mel, wav, sample_name):
        return mel, tf.expand_dims(wav, -1), sample_name


class ForwardPreprocessor:
    def __init__(self, config, tokenizer: Tokenizer):
        self.output_types = (tf.float32, tf.int32, tf.int32, tf.string)
        self.padded_shapes = ([-1, config['mel_channels']], [-1], [-1], [])
        self.padding_values = None
        self.tokenizer = tokenizer
    
    def __call__(self, text, mel, durations, sample_name):
        encoded_phonemes = self.tokenizer(text)
        return mel, encoded_phonemes, durations, sample_name


class AutoregressivePreprocessor:
    
    def __init__(self,
                 config,
                 tokenizer: Tokenizer):
        self.output_types = (tf.float32, tf.int32, tf.int32, tf.string)
        self.padded_shapes = ([-1, config['mel_channels']], [-1], [-1], [])
        self.padding_values = None
        self.start_vec = np.ones((1, config['mel_channels'])) * config['mel_start_value']
        self.end_vec = np.ones((1, config['mel_channels'])) * config['mel_end_value']
        self.tokenizer = tokenizer
    
    def __call__(self, mel, text, sample_name):
        encoded_phonemes = self.tokenizer(text)
        norm_mel = np.concatenate([self.start_vec, mel, self.end_vec], axis=0)
        stop_probs = np.ones((norm_mel.shape[0]))
        stop_probs[-1] = 2
        return norm_mel, encoded_phonemes, stop_probs, sample_name
