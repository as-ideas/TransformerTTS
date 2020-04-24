import os
import shutil
import argparse

import tensorflow as tf
import numpy as np
from tqdm import trange

from utils.config_loader import ConfigLoader
from preprocessing.data_handling import load_files, Dataset, DataPrepper
from utils.decorators import ignore_exception, time_it
from utils.scheduling import piecewise_linear_schedule, reduction_schedule
from utils.logging import SummaryManager

np.random.seed(42)
tf.random.set_seed(42)

# dinamically allocate GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# aux functions declaration

def create_dirs(args):
    base_dir = os.path.join(args.log_dir, session_name)
    log_dir = os.path.join(base_dir, f'logs/')
    weights_dir = os.path.join(base_dir, f'weights/')
    if args.clear_dir:
        delete = input('Delete current logs and weights? (y/[n])')
        if delete == 'y':
            shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    return weights_dir, log_dir, base_dir


@ignore_exception
@time_it
def validate(model,
             val_dataset,
             summary_manager):
    val_loss = {'loss': 0.}
    norm = 0.
    for val_mel, val_text, val_stop in val_dataset.all_batches():
        model_out = model.val_step(inp=val_text,
                                   tar=val_mel,
                                   stop_prob=val_stop)
        norm += 1
        val_loss['loss'] += model_out['loss']
    val_loss['loss'] /= norm
    summary_manager.display_loss(model_out, tag='Validation', plot_all=True)
    summary_manager.display_attention_heads(model_out, tag='ValidationAttentionHeads')
    summary_manager.display_mel(mel=model_out['mel_linear'][0], tag=f'Validation/linear_mel_out')
    summary_manager.display_mel(mel=model_out['final_output'][0], tag=f'Validation/predicted_mel')
    residual = abs(model_out['mel_linear'] - model_out['final_output'])
    summary_manager.display_mel(mel=residual[0], tag=f'Validation/conv-linear_residual')
    summary_manager.display_mel(mel=val_mel[0], tag=f'Validation/target_mel')
    return val_loss['loss']


def print_dict_values(values, key_name, level=0, tab_size=2):
    tab = level * tab_size * ' '
    print(tab + '-', key_name, ':', values)


def print_dictionary(config, recursion_level=0):
    for key in config.keys():
        if isinstance(key, dict):
            recursion_level += 1
            print_dictionary(config[key], recursion_level)
        else:
            print_dict_values(config[key], key_name=key, level=recursion_level)


# consuming CLI, creating paths and directories, load data

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', dest='datadir', type=str)
parser.add_argument('--logdir', dest='log_dir', default='/tmp/summaries', type=str)
parser.add_argument('--config', dest='config', type=str)
parser.add_argument('--cleardir', dest='clear_dir', action='store_true',
                    help="deletes everything under this config's folder.")
parser.add_argument('--session_name', dest='session_name', default=None)
args = parser.parse_args()
session_name = args.session_name
if not session_name:
    session_name = os.path.splitext(os.path.basename(args.config))[0]
config_loader = ConfigLoader(config=args.config)
config_loader.config['datadir'] = args.datadir
config = config_loader.config
weights_paths, log_dir, base_dir = create_dirs(args)
config_loader.dump_config(os.path.join(base_dir, session_name + '.yaml'))

meldir = os.path.join(args.datadir, 'mels')
train_meta = os.path.join(args.datadir, 'train_metafile.txt')
test_meta = os.path.join(args.datadir, 'test_metafile.txt')
train_samples, _ = load_files(metafile=train_meta,
                              meldir=meldir,
                              num_samples=config['n_samples'])  # (phonemes, mel)
val_samples, _ = load_files(metafile=test_meta,
                            meldir=meldir,
                            num_samples=config['n_samples'])  # (phonemes, text, mel)
print('\nCONFIGURATION', session_name)
print_dictionary(config, recursion_level=1)

# get model, prepare data for model, create datasets
model = config_loader.get_model()
config_loader.compile_model(model)
data_prep = DataPrepper(config=config,
                        tokenizer=model.tokenizer)

test_list = [data_prep(s, include_text=True) for s in val_samples]
train_dataset = Dataset(samples=train_samples,
                        preprocessor=data_prep,
                        batch_size=config['batch_size'],
                        shuffle=True)
val_dataset = Dataset(samples=val_samples,
                      preprocessor=data_prep,
                      batch_size=config['batch_size'],
                      shuffle=False)

# create logger and checkpointer and restore latest model

summary_manager = SummaryManager(model=model, log_dir=log_dir, config=config)
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 optimizer=model.optimizer,
                                 net=model)
manager = tf.train.CheckpointManager(checkpoint, weights_paths,
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
_ = train_dataset.next_batch()
t = trange(model.step, config['max_steps'], leave=True)
for _ in t:
    t.set_description(f'step {model.step}')
    mel, phonemes, stop = train_dataset.next_batch()
    decoder_prenet_dropout = piecewise_linear_schedule(model.step, config['dropout_schedule'])
    learning_rate = piecewise_linear_schedule(model.step, config['learning_rate_schedule'])
    reduction_factor = reduction_schedule(model.step, config['reduction_factor_schedule'])
    t.display(f'reduction factor {reduction_factor}', pos=10)
    model.set_constants(decoder_prenet_dropout=decoder_prenet_dropout,
                        learning_rate=learning_rate,
                        reduction_factor=reduction_factor)
    output = model.train_step(inp=phonemes,
                              tar=mel,
                              stop_prob=stop)
    losses.append(float(output['loss']))
    
    t.display(f'step loss: {losses[-1]}', pos=1)
    for pos, n_steps in enumerate(config['n_steps_avg_losses']):
        if len(losses) > n_steps:
            t.display(f'{n_steps}-steps average loss: {sum(losses[-n_steps:]) / n_steps}', pos=pos + 2)
    
    summary_manager.display_loss(output, tag='Train')
    summary_manager.display_scalar(tag='Meta/dropout', scalar_value=decoder_prenet_dropout)
    summary_manager.display_scalar(tag='Meta/learning_rate', scalar_value=model.optimizer.lr)
    summary_manager.display_scalar(tag='Meta/reduction_factor', scalar_value=model.r)
    if (model.step + 1) % config['train_images_plotting_frequency'] == 0:
        summary_manager.display_attention_heads(output, tag='TrainAttentionHeads')
        summary_manager.display_mel(mel=output['mel_linear'][0], tag=f'Train/linear_mel_out')
        summary_manager.display_mel(mel=output['final_output'][0], tag=f'Train/predicted_mel')
        residual = abs(output['mel_linear'] - output['final_output'])
        summary_manager.display_mel(mel=residual[0], tag=f'Train/conv-linear_residual')
        summary_manager.display_mel(mel=mel[0], tag=f'Train/target_mel')
    
    if (model.step + 1) % config['weights_save_frequency'] == 0:
        save_path = manager.save()
        t.display(f'checkpoint at step {model.step}: {save_path}', pos=len(config['n_steps_avg_losses']) + 2)
    
    if (model.step + 1) % config['validation_frequency'] == 0:
        val_loss, time_taken = validate(model=model,
                                        val_dataset=val_dataset,
                                        summary_manager=summary_manager)
        t.display(f'validation loss at step {model.step}: {val_loss} (took {time_taken}s)',
                  pos=len(config['n_steps_avg_losses']) + 3)
    
    if (model.step + 1) % config['prediction_frequency'] == 0 and (model.step >= config['prediction_start_step']):
        for j in range(config['n_predictions']):
            mel, phonemes, stop, text_seq = test_list[j]
            t.display(f'Predicting {j}', pos=len(config['n_steps_avg_losses']) + 4)
            pred = model.predict(phonemes,
                                 max_length=mel.shape[0] + 50,
                                 decoder_prenet_dropout=decoder_prenet_dropout,
                                 encode=False,
                                 verbose=False)
            pred_mel = pred['mel']
            target_mel = mel
            summary_manager.display_attention_heads(outputs=pred, tag=f'TestAttentionHeads/sample {j}')
            summary_manager.display_mel(mel=pred_mel, tag=f'Test/sample {j}/predicted_mel')
            summary_manager.display_mel(mel=target_mel, tag=f'Test/sample {j}/target_mel')
            if model.step > config['audio_start_step']:
                summary_manager.display_audio(tag=f'Target/sample {j}', mel=target_mel)
                summary_manager.display_audio(tag=f'Prediction/sample {j}', mel=pred_mel)

print('Done.')
