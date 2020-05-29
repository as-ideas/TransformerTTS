import argparse
import traceback
import pickle

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.config_manager import ConfigManager
from utils.logging import SummaryManager
from preprocessing.data_handling import load_files, Dataset, DataPrepper
from model.transformer_utils import create_mel_padding_mask
from utils.alignments import get_durations_from_alignment

# dynamically allocate GPU
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

# consuming CLI, creating paths and directories, load data

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', type=str)
parser.add_argument('--session_name', dest='session_name', default=None)
parser.add_argument('--recompute_pred', dest='recompute_pred', action='store_true')
parser.add_argument('--best', dest='best', action='store_true')
parser.add_argument('--binary', dest='binary', action='store_true')
parser.add_argument('--fix_jumps', dest='fix_jumps', action='store_true')
parser.add_argument('--fill_mode_max', dest='fill_mode_max', action='store_true')
parser.add_argument('--fill_mode_next', dest='fill_mode_next', action='store_true')
parser.add_argument('--use_GT', action='store_true')
args = parser.parse_args()
assert (args.fill_mode_max is False) or (args.fill_mode_next is False), 'Choose one gap filling mode.'
weighted = not args.best
binary = args.binary
fill_gaps = args.fill_mode_max or args.fill_mode_next
fix_jumps = args.fix_jumps
fill_mode = f"{f'max' * args.fill_mode_max}{f'next' * args.fill_mode_next}"
filling_tag = f"{f'(max)' * args.fill_mode_max}{f'(next)' * args.fill_mode_next}"
tag_description = ''.join(
    [f'{"_weighted" * weighted}{"_best" * (not weighted)}',
     f'{"_binary" * binary}',
     f'{"_filled" * fill_gaps}{filling_tag}',
     f'{"_fix_jumps" * fix_jumps}'])
writer_tag = f'DurationExtraction{tag_description}'
print(writer_tag)
config_manager = ConfigManager(config_path=args.config, model_kind='autoregressive', session_name=args.session_name)
config = config_manager.config

meldir = config_manager.train_datadir / 'mels'
target_dir = config_manager.train_datadir / f'forward_data'
train_target_dir = target_dir / 'train'
val_target_dir = target_dir / 'val'
train_predictions_dir = target_dir / f'train_predictions_{config_manager.session_name}'
val_predictions_dir = target_dir / f'val_predictions_{config_manager.session_name}'
target_dir.mkdir(exist_ok=True)
train_target_dir.mkdir(exist_ok=True)
val_target_dir.mkdir(exist_ok=True)
train_predictions_dir.mkdir(exist_ok=True)
val_predictions_dir.mkdir(exist_ok=True)
config_manager.dump_config()
script_batch_size = 5 * config['batch_size']
val_has_files = len([batch_file for batch_file in val_predictions_dir.iterdir() if batch_file.suffix == '.npy'])
train_has_files = len([batch_file for batch_file in train_predictions_dir.iterdir() if batch_file.suffix == '.npy'])
model = config_manager.load_model()
if args.recompute_pred or (val_has_files == 0) or (train_has_files == 0):
    train_meta = config_manager.train_datadir / 'train_metafile.txt'
    test_meta = config_manager.train_datadir / 'test_metafile.txt'
    train_samples, _ = load_files(metafile=str(train_meta),
                                  meldir=str(meldir),
                                  num_samples=config['n_samples'])  # (phonemes, mel)
    val_samples, _ = load_files(metafile=str(test_meta),
                                meldir=str(meldir),
                                num_samples=config['n_samples'])  # (phonemes, text, mel)
    
    # get model, prepare data for model, create datasets
    
    data_prep = DataPrepper(config=config,
                            tokenizer=model.tokenizer)
    script_batch_size = 5 * config['batch_size']  # faster parallel computation
    train_dataset = Dataset(samples=train_samples,
                            preprocessor=data_prep,
                            batch_size=script_batch_size,
                            shuffle=False,
                            drop_remainder=False)
    val_dataset = Dataset(samples=val_samples,
                          preprocessor=data_prep,
                          batch_size=script_batch_size,
                          shuffle=False,
                          drop_remainder=False)
    if model.r != 1:
        print(f"ERROR: model's reduction factor is greater than 1, check config. (r={model.r}")
    # identify last decoder block
    n_layers = len(config_manager.config['decoder_num_heads'])
    n_dense = int(config_manager.config['decoder_dense_blocks'])
    n_convs = int(n_layers - n_dense)
    if n_convs > 0:
        last_layer_key = f'Decoder_ConvBlock{n_convs}_CrossfAttention'
    else:
        last_layer_key = f'Decoder_DenseBlock{n_dense}_CrossAttention'
    print(f'Extracting attention from layer {last_layer_key}')
    
    iterator = tqdm(enumerate(val_dataset.all_batches()))
    for c, (val_mel, val_text, val_stop) in iterator:
        iterator.set_description(f'Processing validation set')
        outputs = model.val_step(inp=val_text,
                                 tar=val_mel,
                                 stop_prob=val_stop)
        if args.use_GT:
            batch = (val_mel.numpy(), val_text.numpy(), outputs['decoder_attention'][last_layer_key].numpy())
        else:
            mask = create_mel_padding_mask(val_mel)
            out_val = tf.expand_dims(1 - tf.squeeze(create_mel_padding_mask(val_mel[:, 1:, :])), -1) * outputs[
                'final_output'].numpy()
            batch = (out_val.numpy(), val_text.numpy(), outputs['decoder_attention'][last_layer_key].numpy())
        with open(str(val_predictions_dir / f'{c}_batch_prediction.npy'), 'wb') as file:
            pickle.dump(batch, file)
    
    iterator = tqdm(enumerate(train_dataset.all_batches()))
    for c, (train_mel, train_text, train_stop) in iterator:
        iterator.set_description(f'Processing training set')
        outputs = model.val_step(inp=train_text,
                                 tar=train_mel,
                                 stop_prob=train_stop)
        if args.use_GT:
            batch = (train_mel.numpy(), train_text.numpy(), outputs['decoder_attention'][last_layer_key].numpy())
        else:
            mask = create_mel_padding_mask(train_mel)
            out_train = tf.expand_dims(1 - tf.squeeze(create_mel_padding_mask(train_mel[:, 1:, :])), -1) * outputs[
                'final_output'].numpy()
            batch = (out_train.numpy(), train_text.numpy(), outputs['decoder_attention'][last_layer_key].numpy())
        with open(str(train_predictions_dir / f'{c}_batch_prediction.npy'), 'wb') as file:
            pickle.dump(batch, file)

summary_manager = SummaryManager(model=model, log_dir=config_manager.log_dir / writer_tag, config=config,
                                 default_writer=writer_tag)
val_batch_files = [batch_file for batch_file in val_predictions_dir.iterdir() if batch_file.suffix == '.npy']
iterator = tqdm(enumerate(val_batch_files))
all_val_durations = np.array([])
new_alignments = []
total_val_samples = 0
for c, batch_file in iterator:
    iterator.set_description(f'Extracting validation alignments')
    val_mel, val_text, val_alignments = np.load(str(batch_file), allow_pickle=True)
    durations, unpad_mels, unpad_phonemes, final_align = get_durations_from_alignment(
        batch_alignments=val_alignments,
        mels=val_mel,
        phonemes=val_text,
        weighted=weighted,
        binary=binary,
        fill_gaps=fill_gaps,
        fill_mode=fill_mode,
        fix_jumps=fix_jumps)
    batch_size = len(val_mel)
    for i in range(batch_size):
        sample_idx = total_val_samples + i
        all_val_durations = np.append(all_val_durations, durations[i])
        new_alignments.append(final_align[i])
        sample = (unpad_mels[i], unpad_phonemes[i], durations[i])
        np.save(str(val_target_dir / f'{sample_idx}_mel_phon_dur.npy'), sample)
    total_val_samples += batch_size
all_val_durations[all_val_durations >= 20] = 20
buckets = len(set(all_val_durations))
summary_manager.add_histogram(values=all_val_durations, tag='ValidationDurations', buckets=buckets)
for i, alignment in enumerate(new_alignments):
    summary_manager.add_image(tag='ExtractedValidationAlignments',
                              image=tf.expand_dims(tf.expand_dims(alignment, 0), -1),
                              step=i)

train_batch_files = [batch_file for batch_file in train_predictions_dir.iterdir() if batch_file.suffix == '.npy']
iterator = tqdm(enumerate(train_batch_files))
all_train_durations = np.array([])
new_alignments = []
total_train_samples = 0
for c, batch_file in iterator:
    iterator.set_description(f'Extracting training alignments')
    train_mel, train_text, train_alignments = np.load(str(batch_file), allow_pickle=True)
    durations, unpad_mels, unpad_phonemes, final_align = get_durations_from_alignment(
        batch_alignments=train_alignments,
        mels=train_mel,
        phonemes=train_text,
        weighted=weighted,
        binary=binary,
        fill_gaps=fill_gaps,
        fill_mode=fill_mode,
        fix_jumps=fix_jumps)
    batch_size = len(train_mel)
    for i in range(batch_size):
        sample_idx = total_train_samples + i
        sample = (unpad_mels[i], unpad_phonemes[i], durations[i])
        new_alignments.append(final_align[i])
        all_train_durations = np.append(all_train_durations, durations[i])
        np.save(str(train_target_dir / f'{sample_idx}_mel_phon_dur.npy'), sample)
    total_train_samples += batch_size
all_train_durations[all_train_durations >= 20] = 20
buckets = len(set(all_train_durations))
summary_manager.add_histogram(values=all_train_durations, tag='TrainDurations', buckets=buckets)
for i, alignment in enumerate(new_alignments):
    summary_manager.add_image(tag='ExtractedTrainingAlignments', image=tf.expand_dims(tf.expand_dims(alignment, 0), -1),
                              step=i)
print('Done.')
