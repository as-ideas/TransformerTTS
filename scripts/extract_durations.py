import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

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
logdir = Path(args.logdir)
datadir = Path(args.datadir)
sess_name = logdir.name
config_name = sess_name + '.yaml'
config_loader = ConfigLoader(config=str(logdir / config_name))
config = config_loader.config
meldir = datadir / 'mels'
target_dir = datadir / 'forward_data'
train_target_dir = target_dir / 'train'
val_target_dir = target_dir / 'val'
target_dir.mkdir(exist_ok=True)
train_target_dir.mkdir(exist_ok=True)
val_target_dir.mkdir(exist_ok=True)
config_loader.dump_config(str(target_dir / config_name))
train_meta = datadir / 'train_metafile.txt'
test_meta = datadir / 'test_metafile.txt'
train_samples, _ = load_files(metafile=str(train_meta),
                              meldir=str(meldir),
                              num_samples=config['n_samples'])  # (phonemes, mel)
val_samples, _ = load_files(metafile=str(test_meta),
                            meldir=str(meldir),
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

model.load_checkpoint(str(logdir / 'weights'), r=10)
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
        sample_idx = c * config['batch_size'] + i
        sample = (unpad_mels[i], unpad_phonemes[i], durations[i])
        np.save(str(val_target_dir / f'{sample_idx}_mel_phon_dur.npy'), sample)

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
        sample_idx = c * config['batch_size'] + i
        sample = (unpad_mels[i], unpad_phonemes[i], durations[i])
        np.save(str(train_target_dir / f'{sample_idx}_mel_phon_dur.npy'), sample)
print('Done.')
