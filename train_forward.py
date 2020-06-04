import argparse
import traceback
from pathlib import Path
from time import time

import tensorflow as tf
import numpy as np
from tqdm import trange

from utils.config_manager import ConfigManager
from preprocessing.data_handling import Dataset, ForwardDataPrepper
from utils.decorators import ignore_exception, time_it
from utils.scheduling import piecewise_linear_schedule, reduction_schedule
from utils.logging import SummaryManager
from model.transformer_utils import create_mel_padding_mask

np.random.seed(42)
tf.random.set_seed(42)

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


def build_file_list(data_dir: Path):
    sample_paths = []
    for item in data_dir.iterdir():
        if item.suffix == '.npy':
            sample_paths.append(str(item))
    return sample_paths


@ignore_exception
@time_it
def validate(model,
             val_dataset,
             summary_manager):
    val_loss = {'loss': 0.}
    norm = 0.
    for mel, phonemes, durations in val_dataset.all_batches():
        model_out = model.val_step(input_sequence=phonemes,
                                   target_sequence=mel,
                                   target_durations=durations)
        norm += 1
        val_loss['loss'] += model_out['loss']
    val_loss['loss'] /= norm
    summary_manager.display_loss(model_out, tag='Validation', plot_all=True)
    summary_manager.display_attention_heads(model_out, tag='ValidationAttentionHeads')
    summary_manager.add_histogram(tag=f'Validation/Predicted durations', values=model_out['duration'])
    summary_manager.add_histogram(tag=f'Validation/Target durations', values=durations)
    summary_manager.display_mel(mel=model_out['mel'][0], tag=f'Validation/predicted_mel')
    summary_manager.display_mel(mel=mel[0], tag=f'Validation/target_mel')
    return val_loss['loss']


# consuming CLI, creating paths and directories, load data

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', type=str)
parser.add_argument('--reset_dir', dest='clear_dir', action='store_true',
                    help="deletes everything under this config's folder.")
parser.add_argument('--reset_logs', dest='clear_logs', action='store_true',
                    help="deletes logs under this config's folder.")
parser.add_argument('--reset_weights', dest='clear_weights', action='store_true',
                    help="deletes weights under this config's folder.")
parser.add_argument('--session_name', dest='session_name', default=None)
args = parser.parse_args()

config_manager = ConfigManager(config_path=args.config, model_kind='forward', session_name=args.session_name)
config = config_manager.config
config_manager.create_remove_dirs(clear_dir=args.clear_dir,
                                  clear_logs=args.clear_logs,
                                  clear_weights=args.clear_weights)
config_manager.dump_config()
config_manager.print_config()

train_data_list = build_file_list(config_manager.train_datadir / 'forward_data/train')
dataprep = ForwardDataPrepper()
train_dataset = Dataset(samples=train_data_list,
                        mel_channels=config['mel_channels'],
                        preprocessor=dataprep,
                        batch_size=config['batch_size'],
                        shuffle=True)
val_data_list = build_file_list(config_manager.train_datadir / 'forward_data/val')
val_dataset = Dataset(samples=val_data_list,
                      mel_channels=config['mel_channels'],
                      preprocessor=dataprep,
                      batch_size=config['batch_size'],
                      shuffle=False)

# get model, prepare data for model, create datasets
model = config_manager.get_model()
config_manager.compile_model(model)

# create logger and checkpointer and restore latest model
summary_manager = SummaryManager(model=model, log_dir=config_manager.log_dir, config=config)
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 optimizer=model.optimizer,
                                 net=model)
manager = tf.train.CheckpointManager(checkpoint, config_manager.weights_dir,
                                     max_to_keep=config['keep_n_weights'],
                                     keep_checkpoint_every_n_hours=config['keep_checkpoint_every_n_hours'])
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print(f'\nresuming training from step {model.step} ({manager.latest_checkpoint})')
else:
    print(f'\nstarting training from scratch')
# main event
print('\nTRAINING')
losses = []
test_batch = val_dataset.next_batch()
t = trange(model.step, config['max_steps'], leave=True)
for _ in t:
    t.set_description(f'step {model.step}')
    mel, phonemes, durations = train_dataset.next_batch()
    learning_rate = piecewise_linear_schedule(model.step, config['learning_rate_schedule'])
    decoder_prenet_dropout = piecewise_linear_schedule(model.step, config['decoder_dropout_schedule'])
    drop_n_heads = tf.cast(reduction_schedule(model.step, config['head_drop_schedule']), tf.int32)
    model.set_constants(decoder_prenet_dropout=decoder_prenet_dropout,
                        learning_rate=learning_rate,
                        drop_n_heads=drop_n_heads)
    output = model.train_step(input_sequence=phonemes,
                              target_sequence=mel,
                              target_durations=durations)
    losses.append(float(output['loss']))
    
    t.display(f'step loss: {losses[-1]}', pos=1)
    for pos, n_steps in enumerate(config['n_steps_avg_losses']):
        if len(losses) > n_steps:
            t.display(f'{n_steps}-steps average loss: {sum(losses[-n_steps:]) / n_steps}', pos=pos + 2)
    
    summary_manager.display_loss(output, tag='Train')
    summary_manager.display_scalar(tag='Meta/learning_rate', scalar_value=model.optimizer.lr)
    summary_manager.display_scalar(tag='Meta/decoder_prenet_dropout', scalar_value=model.decoder_prenet.rate)
    summary_manager.display_scalar(tag='Meta/drop_n_heads', scalar_value=model.drop_n_heads)
    if model.step % config['train_images_plotting_frequency'] == 0:
        summary_manager.display_attention_heads(output, tag='TrainAttentionHeads')
        summary_manager.display_mel(mel=output['mel'][0], tag=f'Train/predicted_mel')
        summary_manager.display_mel(mel=mel[0], tag=f'Train/target_mel')
        summary_manager.add_histogram(tag=f'Train/Predicted durations', values=output['duration'])
        summary_manager.add_histogram(tag=f'Train/Target durations', values=durations)
    
    if model.step % config['weights_save_frequency'] == 0:
        save_path = manager.save()
        t.display(f'checkpoint at step {model.step}: {save_path}', pos=len(config['n_steps_avg_losses']) + 2)
    
    if model.step % config['validation_frequency'] == 0:
        t.display(f'Validating', pos=len(config['n_steps_avg_losses']) + 3)
        val_loss, time_taken = validate(model=model,
                                        val_dataset=val_dataset,
                                        summary_manager=summary_manager)
        t.display(f'validation loss at step {model.step}: {val_loss} (took {time_taken}s)',
                  pos=len(config['n_steps_avg_losses']) + 3)
    
    if model.step % config['prediction_frequency'] == 0 and (model.step >= config['prediction_start_step']):
        tar_mel, phonemes, durs = test_batch
        t.display(f'Predicting', pos=len(config['n_steps_avg_losses']) + 4)
        timed_pred = time_it(model.predict)
        model_out, time_taken = timed_pred(phonemes, encode=False)
        summary_manager.display_attention_heads(model_out, tag='TestAttentionHeads')
        summary_manager.add_histogram(tag=f'Test/Predicted durations', values=model_out['duration'])
        summary_manager.add_histogram(tag=f'Test/Target durations', values=durs)
        pred_lengths = tf.cast(tf.reduce_sum(1 - model_out['expanded_mask'], axis=-1), tf.int32)
        pred_lengths = tf.squeeze(pred_lengths)
        tar_lengths = tf.cast(tf.reduce_sum(1 - create_mel_padding_mask(tar_mel), axis=-1), tf.int32)
        tar_lengths = tf.squeeze(tar_lengths)
        display_start = time()
        for j, pred_mel in enumerate(model_out['mel']):
            predval = pred_mel[:pred_lengths[j], :]
            tar_value = tar_mel[j, :tar_lengths[j], :]
            summary_manager.display_mel(mel=predval, tag=f'Test/sample {j}/predicted_mel')
            summary_manager.display_mel(mel=tar_value, tag=f'Test/sample {j}/target_mel')
            if j < config['n_predictions']:
                if model.step >= config['audio_start_step'] and (
                        model.step % config['audio_prediction_frequency'] == 0):
                    summary_manager.display_audio(tag=f'Target/sample {j}', mel=tar_value)
                    summary_manager.display_audio(tag=f'Prediction/sample {j}', mel=predval)
            else:
                break
        display_end = time()
        t.display(f'Predictions took {time_taken}. Displaying took {display_end - display_start}.',
                  pos=len(config['n_steps_avg_losses']) + 4)
print('Done.')
