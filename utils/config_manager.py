import subprocess
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import ruamel.yaml

from model.models import AutoregressiveTransformer, ForwardTransformer


class ConfigManager:
    
    def __init__(self, config_path: str, model_kind: str, session_name: str = None):
        if model_kind not in ['autoregressive', 'forward']:
            raise TypeError(f"model_kind must be in {['autoregressive', 'forward']}")
        self.config_path = Path(config_path)
        self.model_kind = model_kind
        self.yaml = ruamel.yaml.YAML()
        self.config, self.data_config, self.model_config = self._load_config()
        self.git_hash = self._get_git_hash()
        if session_name is None:
            if self.config['session_name'] is None:
                session_name = self.git_hash
        self.session_name = '_'.join(filter(None, [self.config_path.name, session_name]))
        self.base_dir, self.log_dir, self.train_datadir, self.weights_dir = self._make_folder_paths()
        self.learning_rate = np.array(self.config['learning_rate_schedule'])[0, 1].astype(np.float32)
        self.max_r = np.array(self.config['reduction_factor_schedule'])[0, 1].astype(np.int32)
        if model_kind == 'autoregressive':
            self.stop_scaling = self.config.get('stop_loss_scaling', 1.)
    
    def _load_config(self):
        data_config = self.yaml.load(open(str(self.config_path / 'data_config.yaml'), 'rb'))
        model_config = self.yaml.load(open(str(self.config_path / f'{self.model_kind}_config.yaml'), 'rb'))
        all_config = {}
        all_config.update(model_config)
        all_config.update(data_config)
        return all_config, data_config, model_config
    
    @staticmethod
    def _get_git_hash():
        try:
            return subprocess.check_output(["git", "describe", "--always"]).strip().decode()
        except Exception as e:
            print(f"WARNING: could not retrieve git hash. {e}")
    
    def _check_hash(self):
        try:
            git_hash = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
            if self.config['git_hash'] != git_hash:
                print(f"WARNING: git hash mismatch. Current: {git_hash}. Config hash: {self.config['git_hash']}")
        except Exception as e:
            print(f"WARNING: could not check git hash. {e}")
    
    def _make_folder_paths(self):
        base_dir = Path(self.config['log_directory']) / self.session_name
        log_dir = base_dir / f'{self.model_kind}_logs'
        weights_dir = base_dir / f'{self.model_kind}_weights'
        train_datadir = self.config['train_data_directory']
        if train_datadir is None:
            train_datadir = self.config['data_directory']
        train_datadir = Path(train_datadir)
        return base_dir, log_dir, train_datadir, weights_dir
    
    @staticmethod
    def _print_dict_values(values, key_name, level=0, tab_size=2):
        tab = level * tab_size * ' '
        print(tab + '-', key_name, ':', values)
    
    def _print_dictionary(self, dictionary, recursion_level=0):
        for key in dictionary.keys():
            if isinstance(key, dict):
                recursion_level += 1
                self._print_dictionary(dictionary[key], recursion_level)
            else:
                self._print_dict_values(dictionary[key], key_name=key, level=recursion_level)
    
    def print_config(self):
        print('\nCONFIGURATION', self.session_name)
        self._print_dictionary(self.config)
    
    def update_config(self):
        self.config['git_hash'] = self.git_hash
        self.model_config['git_hash'] = self.git_hash
        self.data_config['session_name'] = self.session_name
        self.model_config['session_name'] = self.session_name
        self.config['session_name'] = self.session_name
    
    def get_model(self, ignore_hash=False):
        if not ignore_hash:
            self._check_hash()
        return AutoregressiveTransformer(mel_channels=self.config['mel_channels'],
                                         encoder_model_dimension=self.config['encoder_model_dimension'],
                                         decoder_model_dimension=self.config['decoder_model_dimension'],
                                         encoder_num_heads=self.config['encoder_num_heads'],
                                         decoder_num_heads=self.config['decoder_num_heads'],
                                         encoder_feed_forward_dimension=self.config['encoder_feed_forward_dimension'],
                                         decoder_feed_forward_dimension=self.config['decoder_feed_forward_dimension'],
                                         encoder_maximum_position_encoding=self.config['encoder_max_position_encoding'],
                                         decoder_maximum_position_encoding=self.config['decoder_max_position_encoding'],
                                         encoder_dense_blocks=self.config['encoder_dense_blocks'],
                                         decoder_dense_blocks=self.config['decoder_dense_blocks'],
                                         decoder_prenet_dimension=self.config['decoder_prenet_dimension'],
                                         encoder_prenet_dimension=self.config['encoder_prenet_dimension'],
                                         postnet_conv_filters=self.config['postnet_conv_filters'],
                                         postnet_conv_layers=self.config['postnet_conv_layers'],
                                         postnet_kernel_size=self.config['postnet_kernel_size'],
                                         dropout_rate=self.config['dropout_rate'],
                                         max_r=self.max_r,
                                         mel_start_value=self.config['mel_start_value'],
                                         mel_end_value=self.config['mel_end_value'],
                                         phoneme_language=self.config['phoneme_language'],
                                         debug=self.config['debug'])
    
    def get_forward_model(self, ignore_hash=False):
        if not ignore_hash:
            self._check_hash()
        return ForwardTransformer(model_dim=self.config['model_dimension'],
                                  dropout_rate=self.config['dropout_rate'],
                                  decoder_num_heads=self.config['decoder_num_heads'],
                                  encoder_num_heads=self.config['encoder_num_heads'],
                                  encoder_maximum_position_encoding=self.config['encoder_max_position_encoding'],
                                  decoder_maximum_position_encoding=self.config['decoder_max_position_encoding'],
                                  encoder_feed_forward_dimension=self.config['encoder_feed_forward_dimension'],
                                  decoder_feed_forward_dimension=self.config['decoder_feed_forward_dimension'],
                                  mel_channels=self.config['mel_channels'],
                                  encoder_dense_blocks=self.config['encoder_dense_blocks'],
                                  decoder_dense_blocks=self.config['decoder_dense_blocks'],
                                  phoneme_language=self.config['phoneme_language'],
                                  debug=self.config['debug'], )
    
    def compile_model(self, model):
        model._compile(stop_scaling=self.stop_scaling, optimizer=self.new_adam(self.learning_rate))
    
    def compile_forward_model(self, model):
        model._compile(optimizer=self.new_adam(self.learning_rate))
    
    # TODO: move to model
    @staticmethod
    def new_adam(learning_rate):
        return tf.keras.optimizers.Adam(learning_rate,
                                        beta_1=0.9,
                                        beta_2=0.98,
                                        epsilon=1e-9)
    
    def dump_config(self, log_dir: str):
        self.update_config()
        log_dir = Path(log_dir) / 'config'
        log_dir.mkdir(exist_ok=True)
        self.yaml.dump(self.model_config, open(log_dir / f'{self.model_kind}_config.yaml', 'w'))
        self.yaml.dump(self.data_config, open(log_dir / 'data_config.yaml', 'w'))
    
    def create_remove_dirs(self, clear_dir: False, clear_logs: False, clear_weights: False):
        self.base_dir.mkdir(exist_ok=True)
        if clear_dir:
            delete = input(f'Delete {self.log_dir} AND {self.weights_dir}? (y/[n])')
            if delete == 'y':
                shutil.rmtree(self.log_dir, ignore_errors=True)
                shutil.rmtree(self.weights_dir, ignore_errors=True)
        if clear_logs:
            delete = input(f'Delete {self.log_dir}? (y/[n])')
            if delete == 'y':
                shutil.rmtree(self.log_dir, ignore_errors=True)
        if clear_weights:
            delete = input(f'Delete {self.weights_dir}? (y/[n])')
            if delete == 'y':
                shutil.rmtree(self.weights_dir, ignore_errors=True)
        self.log_dir.mkdir(exist_ok=True)
        self.weights_dir.mkdir(exist_ok=True)
