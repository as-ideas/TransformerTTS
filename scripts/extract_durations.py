import os
import argparse

import numpy as np
from tqdm import trange, tqdm

from utils.config_loader import ConfigLoader
from preprocessing.data_handling import load_files, Dataset, DataPrepper
from utils.scheduling import piecewise_linear_schedule, reduction_schedule
from utils.alignments import get_durations_from_alignment

# consuming CLI, creating paths and directories, load data

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', dest='datadir', type=str)
parser.add_argument('--logdir', dest='logdir', type=str)
parser.add_argument('--reset_dir', dest='clear_dir', action='store_true',
                    help="deletes everything under this config's folder.")
args = parser.parse_args()
sess_name = os.path.basename(args.logdir)
config_loader = ConfigLoader(config=os.path.join(args.logdir, sess_name + '.yaml'))
config = config_loader.config
meldir = os.path.join(args.datadir, 'mels')
target_dir = os.path.join(args.datadir, 'forward_data')
config_loader.dump_config(os.path.join(target_dir, sess_name + '.yaml'))
train_target_dir = os.path.join(target_dir, 'train')
val_target_dir = os.path.join(target_dir, 'val')
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
if not os.path.exists(train_target_dir):
    os.mkdir(train_target_dir)
if not os.path.exists(val_target_dir):
    os.mkdir(val_target_dir)
train_meta = os.path.join(args.datadir, 'train_metafile.txt')
test_meta = os.path.join(args.datadir, 'test_metafile.txt')
train_samples, _ = load_files(metafile=train_meta,
                              meldir=meldir,
                              num_samples=config['n_samples'])  # (phonemes, mel)
val_samples, _ = load_files(metafile=test_meta,
                            meldir=meldir,
                            num_samples=config['n_samples'])  # (phonemes, text, mel)

# get model, prepare data for model, create datasets
model = config_loader.get_model()
config_loader.compile_model(model)
data_prep = DataPrepper(config=config,
                        tokenizer=model.tokenizer)
train_dataset = Dataset(samples=train_samples,
                        preprocessor=data_prep,
                        batch_size=config['batch_size'],
                        shuffle=False,
                        drop_remainder=False)
val_dataset = Dataset(samples=val_samples,
                      preprocessor=data_prep,
                      batch_size=config['batch_size'],
                      shuffle=False,
                      drop_remainder=False)

model.load_checkpoint(args.logdir + '/weights', r=10)
decoder_prenet_dropout = piecewise_linear_schedule(model.step, config['dropout_schedule'])
reduction_factor = reduction_schedule(model.step, config['reduction_factor_schedule'])
model.set_constants(decoder_prenet_dropout=decoder_prenet_dropout,
                    reduction_factor=reduction_factor)

# 'log_dir': tf.summary.create_file_writer(str(self.log_dir)) #TODO: log plots in TB
iterator = tqdm(enumerate(val_dataset.all_batches()))
for c, (val_mel, val_text, val_stop) in iterator:
    iterator.set_description(f'Processing validation set')
    outputs = model.val_step(inp=val_text,
                             tar=val_mel,
                             stop_prob=val_stop)
    durations, unpad_mels, unpad_phonemes = get_durations_from_alignment(
        batch_alignments=outputs['attention_weights']['decoder_layer4_block2'].numpy(),
        mels=val_mel.numpy(),
        phonemes=val_text.numpy(),
        weighted=True,
        PLOT_OUTLIERS=False,
        PLOT_ALL=False)
    for i in range(len(val_mel)):
        sample_idx = c*config['batch_size'] + i
        sample = (unpad_mels[i], unpad_phonemes[i], durations[i])
        np.save(os.path.join(val_target_dir, f'{sample_idx}_mel_phon_dur.npy'), sample)

iterator = tqdm(enumerate(train_dataset.all_batches()))
for c, (train_mel, train_text, train_stop) in iterator:
    iterator.set_description(f'Processing training set')
    outputs = model.val_step(inp=train_text,
                             tar=train_mel,
                             stop_prob=train_stop)
    durations, unpad_mels, unpad_phonemes = get_durations_from_alignment(
        batch_alignments=outputs['attention_weights']['decoder_layer4_block2'].numpy(),
        mels=train_mel.numpy(),
        phonemes=train_text.numpy(),
        weighted=True,
        PLOT_OUTLIERS=False,
        PLOT_ALL=False)
    for i in range(len(train_mel)):
        sample_idx = c*config['batch_size'] + i
        sample = (unpad_mels[i], unpad_phonemes[i], durations[i])
        np.save(os.path.join(train_target_dir, f'{sample_idx}_mel_phon_dur.npy'), sample)
print('Done.')
