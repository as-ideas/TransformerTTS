from pathlib import Path

import tensorflow as tf
import ruamel.yaml

from model.models import ForwardTransformer, Aligner


def tts_ljspeech(version='v1') -> [tf.keras.models.Model, dict]:
    config_name = f'ljspeech_tts_config_{version}.yaml'
    weights_name = f'ljspeech_tts_weights_{version}.hdf5'
    remote_dir = Path(f'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/api_weights/')
    config_path = tf.keras.utils.get_file(config_name, str(remote_dir / config_name))
    weights_path = tf.keras.utils.get_file(weights_name, str(remote_dir / weights_name))
    return tts_custom(config_path, weights_path)


def tts_custom(config_path: str, weights_path: str) -> [tf.keras.models.Model, dict]:
    yaml = ruamel.yaml.YAML()
    with open(config_path, 'rb') as session_yaml:
        config = yaml.load(session_yaml)
    model = ForwardTransformer.from_config(config)
    model.build_model_weights(weights_path)
    return model, config


def aligner_custom(config_path: str, weights_path: str) -> [tf.keras.models.Model, dict]:
    yaml = ruamel.yaml.YAML()
    with open(config_path, 'rb') as session_yaml:
        config = yaml.load(session_yaml)
    model = Aligner.from_config(config)
    model.build_model_weights(weights_path)
    return model, config
