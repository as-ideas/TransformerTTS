import argparse

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.config_manager import Config
from utils.logging_utils import SummaryManager
from preprocessing.datasets import AutoregressivePreprocessor
from model.transformer_utils import create_mel_padding_mask
from utils.alignments import get_durations_from_alignment
from utils.scripts_utils import dynamic_memory_allocation
from preprocessing.datasets import TextMelDataset

np.random.seed(42)
tf.random.set_seed(42)
dynamic_memory_allocation()

# consuming CLI, creating paths and directories, load data

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', type=str)
parser.add_argument('--recompute_pred', dest='recompute_pred', action='store_true',
                    help='Recompute the model predictions.')
parser.add_argument('--best', dest='best', action='store_true',
                    help='Use best head instead of weighted average of heads.')
parser.add_argument('--binary', dest='binary', action='store_true',
                    help='Use attention peak instead of all attentio values.')
parser.add_argument('--fix_jumps', dest='fix_jumps', action='store_true',
                    help='Scan attention peaks and try to fix jumps. Only with binary.')
parser.add_argument('--fill_mode_max', dest='fill_mode_max', action='store_true',
                    help='Fill zero durations with ones. Reduces phoneme duration with maximum value in sequence to compensate.')
parser.add_argument('--fill_mode_next', dest='fill_mode_next', action='store_true',
                    help='Fill zero durations with ones. Reduces next non-zero phoneme duration in sequence to compensate.')
parser.add_argument('--use_GT', action='store_true',
                    help='Use ground truth mel instead of predicted mel to train forward model.')
parser.add_argument('--autoregressive_weights', type=str, default='',
                    help='Explicit path to autoregressive model weights.')
parser.add_argument('--extract_from_layer', dest='extract_layer', type=int, default=-1)
args = parser.parse_args()
assert (args.fill_mode_max is False) or (args.fill_mode_next is False), 'Choose one gap filling mode.'
weighted = not args.best
binary = args.binary
fill_gaps = args.fill_mode_max or args.fill_mode_next
fix_jumps = args.fix_jumps
fill_mode = f"{f'max' * args.fill_mode_max}{f'next' * args.fill_mode_next}"
filling_tag = f"{f'(max)' * args.fill_mode_max}{f'(next)' * args.fill_mode_next}"
tag_description = ''.join(
    [
        f'{"_weighted" * weighted}{"_best" * (not weighted)}',
        f'{"_binary" * binary}',
        f'{"_filled" * fill_gaps}{filling_tag}',
        f'{"_fix_jumps" * fix_jumps}',
        f'_layer{args.extract_layer}'
    ])
writer_tag = f'DurationExtraction{tag_description}'
print(writer_tag)
config_manager = Config(config_path=args.config, model_kind='autoregressive')
config = config_manager.config
config_manager.print_config()
if args.autoregressive_weights != '':
    model = config_manager.load_model(args.autoregressive_weights)
else:
    model = config_manager.load_model()
if model.r != 1:
    print(f"ERROR: model's reduction factor is greater than 1, check config. (r={model.r}")

data_prep = AutoregressivePreprocessor.from_config(config=config_manager,
                                                   tokenizer=model.text_pipeline.tokenizer)
data_handler = TextMelDataset.from_config(config_manager,
                                          preprocessor=data_prep,
                                          kind='phonemized')

target_dir = config_manager.data_dir / f'durations'
target_dir.mkdir(exist_ok=True)
config_manager.dump_config()
dataset = data_handler.get_dataset(bucket_batch_sizes=[64, 42, 32, 25, 21, 18, 16, 14, 12, 11, 1], shuffle=False, drop_remainder=False)

# identify last decoder block
n_layers = len(config_manager.config['decoder_num_heads'])
n_dense = int(config_manager.config['decoder_dense_blocks'])
n_convs = int(n_layers - n_dense)
if args.extract_layer > 0:
    if n_convs > 0:
        last_layer_key = f'Decoder_ConvBlock{args.extract_layer}_CrossAttention'
    else:
        last_layer_key = f'Decoder_DenseBlock{args.extract_layer}_CrossAttention'
else:
    if n_convs > 0:
        last_layer_key = f'Decoder_ConvBlock{n_convs}_CrossAttention'
    else:
        last_layer_key = f'Decoder_DenseBlock{n_dense}_CrossAttention'
print(f'Extracting attention from layer {last_layer_key}')

summary_manager = SummaryManager(model=model, log_dir=config_manager.log_dir / writer_tag, config=config,
                                 default_writer=writer_tag)
all_durations = np.array([])
new_alignments = []
iterator = tqdm(enumerate(dataset.all_batches()))
step = 0
for c, (mel_batch, text_batch, stop_batch, file_name_batch) in iterator:
    iterator.set_description(f'Processing dataset')
    outputs = model.val_step(inp=text_batch,
                             tar=mel_batch,
                             stop_prob=stop_batch)
    attention_values = outputs['decoder_attention'][last_layer_key].numpy()
    text = text_batch.numpy()
    
    if args.use_GT:
        mel = mel_batch.numpy()
    else:
        pred_mel = outputs['final_output'].numpy()
        mask = create_mel_padding_mask(mel_batch)
        pred_mel = tf.expand_dims(1 - tf.squeeze(create_mel_padding_mask(mel_batch[:, 1:, :])), -1) * pred_mel
        mel = pred_mel.numpy()
    
    durations, final_align, jumpiness, peakiness, diag_measure = get_durations_from_alignment(
        batch_alignments=attention_values,
        mels=mel,
        phonemes=text,
        weighted=weighted,
        binary=binary,
        fill_gaps=fill_gaps,
        fill_mode=fill_mode,
        fix_jumps=fix_jumps)
    batch_avg_jumpiness = tf.reduce_mean(jumpiness, axis=0)
    batch_avg_peakiness = tf.reduce_mean(peakiness, axis=0)
    batch_avg_diag_measure = tf.reduce_mean(diag_measure, axis=0)
    for i in range(tf.shape(jumpiness)[1]):
        summary_manager.display_scalar(tag=f'TrainDecoderAttentionJumpiness/head{i}',
                                       scalar_value=tf.reduce_mean(batch_avg_jumpiness[i]), step=c)
        summary_manager.display_scalar(tag=f'TrainDecoderAttentionPeakiness/head{i}',
                                       scalar_value=tf.reduce_mean(batch_avg_peakiness[i]), step=c)
        summary_manager.display_scalar(tag=f'TrainDecoderAttentionDiagonality/head{i}',
                                       scalar_value=tf.reduce_mean(batch_avg_diag_measure[i]), step=c)
        
    for i, name in enumerate(file_name_batch):
        all_durations = np.append(all_durations, durations[i])  # for plotting only
        summary_manager.add_image(tag='ExtractedAlignments', image=tf.expand_dims(tf.expand_dims(final_align[i], 0), -1),
                                  step=step)
        
        step += 1
        np.save(str(target_dir / f"{name.numpy().decode('utf-8')}.npy"), durations[i])

all_durations[all_durations >= 20] = 20  # for plotting only
buckets = len(set(all_durations))  # for plotting only
summary_manager.add_histogram(values=all_durations, tag='ExtractedDurations', buckets=buckets)
print('Done.')
