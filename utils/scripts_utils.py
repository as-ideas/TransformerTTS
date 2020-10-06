import traceback
import argparse

import tensorflow as tf


def dynamic_memory_allocation():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
        except Exception:
            traceback.print_exc()


def basic_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', type=str)
    parser.add_argument('--reset_dir', dest='clear_dir', action='store_true',
                        help="deletes everything under this config's folder.")
    parser.add_argument('--reset_logs', dest='clear_logs', action='store_true',
                        help="deletes logs under this config's folder.")
    parser.add_argument('--reset_weights', dest='clear_weights', action='store_true',
                        help="deletes weights under this config's folder.")
    return parser
