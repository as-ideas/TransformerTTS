from pathlib import Path
import subprocess

import numpy as np
import tensorflow as tf
import ruamel.yaml

from model.models import AutoregressiveTransformer, ForwardTransformer


class ConfigLoader:
    
    def __init__(self, config_path: str, model_kind: str):
        if model_kind not in ['autoregressive', 'forward']:
            raise TypeError(f"model_kind must be in {['autoregressive', 'forward']}")
        self.model_kind = model_kind
        self.yaml = ruamel.yaml.YAML()
        self.config, self.data_config, self.model_config = self._load_config(Path(config_path))
        self.learning_rate = np.array(self.config['learning_rate_schedule'])[0, 1].astype(np.float32)
        self.max_r = np.array(self.config['reduction_factor_schedule'])[0, 1].astype(np.int32)
        if model_kind == 'autoregressive':
            self.stop_scaling = self.config.get('stop_loss_scaling', 1.)
    
    def _load_config(self, config_path: Path):
        data_config = self.yaml.load(open(str(config_path / 'data_config.yaml'), 'rb'))
        model_config = self.yaml.load(open(str(config_path / f'{self.model_kind}_config.yaml'), 'rb'))
        all_config = {}
        all_config.update(model_config)
        all_config.update(data_config)
        return all_config, data_config, model_config
    
    def update_config(self, data_dir):
        git_hash = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
        self.config['datadir'] = data_dir
        self.config['git_hash'] = git_hash
        self.data_config['datadir'] = data_dir
        self.model_config['git_hash'] = git_hash
    
    def _check_hash(self):
        try:
            git_hash = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
            if self.config['git_hash'] != git_hash:
                print(f"WARNING: git hash mismatch. Current: {git_hash}. Config hash: {self.config['git_hash']}")
        except Exception as e:
            print(f"WARNING: could not check git hash. {e}")
    
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
        log_dir = Path(log_dir) / 'config'
        log_dir.mkdir(exist_ok=True)
        self.yaml.dump(self.model_config, open(log_dir / f'{self.model_kind}_config.yaml', 'w'))
        self.yaml.dump(self.data_config, open(log_dir / 'data_config.yaml', 'w'))
