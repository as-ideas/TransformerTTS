import os
import shutil
import argparse

import ruamel.yaml
import tensorflow as tf
import numpy as np
from tqdm import trange

from model.combiner import Combiner
from preprocessing.data_handling import load_files, Dataset
from preprocessing.preprocessor import DataPrepper
from utils.decorators import ignore_exception, time_it
from utils.scheduling import piecewise_linear_schedule, reduction_schedule
from utils.logging import SummaryManager

np.random.seed(42)
tf.random.set_seed(42)


# aux functions declaration

def create_dirs(args):
    base_dir = os.path.join(args.log_dir, config_name)
    log_dir = os.path.join(base_dir, f'logs/')
    weights_dir = os.path.join(base_dir, f'weights/')
    if args.clear_dir:
        shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    return weights_dir, log_dir, base_dir


@ignore_exception
@time_it
def validate(combiner,
             val_dataset,
             summary_manager,
             decoder_prenet_dropout):
    val_loss = {'loss': 0.}
    norm = 0.
    for val_mel, val_text, val_stop in val_dataset.all_batches():
        model_out = combiner.val_step(val_text,
                                      val_mel,
                                      val_stop,
                                      pre_dropout=decoder_prenet_dropout)
        norm += 1
        val_loss['loss'] += model_out['loss']
    val_loss['loss'] /= norm
    summary_manager.display_loss(model_out, tag='Validation', plot_all=True)
    summary_manager.display_attention_heads(model_out, tag='Validation')
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
parser.add_argument('--cleardir', dest='clear_dir', action='store_true', help="deletes everything under this config's folder.")
args = parser.parse_args()
yaml = ruamel.yaml.YAML()
config = yaml.load(open(args.config, 'r'))
config_name = os.path.splitext(os.path.basename(args.config))[0]
config['datadir'] = args.datadir
weights_paths, log_dir, base_dir = create_dirs(args)

print('creating model')
if not config['use_phonemes'] == True:
    print('Set use_phonemes: True in config')
    exit()
meldir = os.path.join(args.datadir, 'mels')
train_meta = os.path.join(args.datadir, 'train_metafile.txt')
test_meta = os.path.join(args.datadir, 'test_metafile.txt')

train_samples, _ = load_files(metafile=train_meta,
                              meldir=meldir,
                              num_samples=config['n_samples'])
# OUT = (phonemes, text, mel)
val_samples, _ = load_files(metafile=test_meta,
                            meldir=meldir,
                            num_samples=config['n_samples'])
print('\nCONFIGURATION', config_name)
print_dictionary(config, recursion_level=1)

# get model, prepare data for model, create datasets

combiner = Combiner(config=config)
data_prep = DataPrepper(mel_channels=config['mel_channels'],
                        start_vec_val=config['mel_start_vec_value'],
                        end_vec_val=config['mel_end_vec_value'],
                        tokenizer=combiner.tokenizer)
yaml.dump(config, open(os.path.join(base_dir, os.path.basename(args.config)), 'w'))
test_list = [data_prep(s, include_text=True) for s in val_samples]
train_dataset = Dataset(samples=train_samples,
                        preprocessor=data_prep,
                        batch_size=config['batch_size'],
                        shuffle=True)
val_dataset = Dataset(samples=val_samples,
                      preprocessor=data_prep,
                      batch_size=config['batch_size'],
                      shuffle=False)
# val_list = [data_prep(s) for s in val_samples]

# create logger and checkpointer and restore latest model

summary_manager = SummaryManager(combiner=combiner, log_dir=log_dir)
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 optimizer=combiner.text_mel.optimizer,
                                 net=combiner.text_mel)
manager = tf.train.CheckpointManager(checkpoint, weights_paths,
                                     max_to_keep=config['keep_n_weights'],
                                     keep_checkpoint_every_n_hours=config['keep_checkpoint_every_n_hours'])
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print(f'\nresuming training from step {combiner.step} ({manager.latest_checkpoint})')
else:
    print(f'\nstarting training from scratch')

# main event

print('\nTRAINING')
losses = []
_ = train_dataset.next_batch()
t = trange(combiner.step, config['max_steps'], leave=True)
for i in t:
    t.set_description(f'step {combiner.step}')
    mel, phonemes, stop = train_dataset.next_batch()
    decoder_prenet_dropout = piecewise_linear_schedule(combiner.step, config['dropout_schedule'])
    learning_rate = piecewise_linear_schedule(combiner.step, config['learning_rate_schedule'])
    reduction_factor = reduction_schedule(combiner.step, config['reduction_schedule'])
    t.display(f'red fac {reduction_factor}', pos=10)
    combiner.text_mel.set_r(reduction_factor)
    combiner.set_learning_rates(learning_rate)
    
    output = combiner.train_step(text=phonemes,
                                 mel=mel,
                                 stop=stop,
                                 pre_dropout=decoder_prenet_dropout)
    
    losses.append(float(output['loss']))
    
    t.display(f'step loss: {losses[-1]}', pos=1)
    for pos, n_steps in enumerate(config['n_steps_avg_losses']):
        if len(losses) > n_steps:
            t.display(f'{n_steps}-steps average loss: {sum(losses[-n_steps:]) / n_steps}', pos=pos + 2)
    
    summary_manager.display_loss(output, tag='Train')
    summary_manager.display_scalar(tag='Meta/dropout', scalar_value=decoder_prenet_dropout)
    summary_manager.display_scalar(tag='Meta/learning_rate', scalar_value=combiner.text_mel.optimizer.lr)
    if (combiner.step + 1) % config['plot_attention_freq'] == 0:
        summary_manager.display_attention_heads(output, tag='Train')
    
    if (combiner.step + 1) % config['weights_save_freq'] == 0:
        save_path = manager.save()
        t.display(f'checkpoint at step {combiner.step}: {save_path}', pos=len(config['n_steps_avg_losses']) + 2)
    
    if (combiner.step + 1) % config['val_freq'] == 0:
        val_loss, time_taken = validate(combiner=combiner,
                                        val_dataset=val_dataset,
                                        summary_manager=summary_manager,
                                        decoder_prenet_dropout=decoder_prenet_dropout)
        t.display(f'validation loss at step {combiner.step}: {val_loss} (took {time_taken}s)',
                  pos=len(config['n_steps_avg_losses']) + 3)
    
    if (combiner.step + 1) % config['image_freq'] == 0 and (combiner.step > config['prediction_start_step']):
        timings = []
        for i in range(config['n_predictions']):
            mel, phonemes, stop, text_seq = test_list[i]
            t.display(f'Predicting {i}', pos=len(config['n_steps_avg_losses']) + 4)
            pred, time_taken = combiner.predict(phonemes,
                                                pre_dropout=decoder_prenet_dropout,
                                                max_len_mel=mel.shape[0] + 50,
                                                verbose=False)
            timings.append(time_taken)
            summary_manager.display_attention_heads(outputs=pred, tag='Test')
            summary_manager.display_mel(mel=pred['mel'], tag=f'Test/predicted_mel {i}')
            summary_manager.display_mel(mel=mel, tag=f'Test/target_mel {i}')
            if combiner.step > config['audio_start_step']:
                summary_manager.display_audio(tag='Target', mel=mel, config=config)
                summary_manager.display_audio(tag='Prediction', mel=pred['mel'], config=config)
        
        t.display(f"Predictions at time step {combiner.step} took {sum(timings)}s ({timings})",
                  pos=len(config['n_steps_avg_losses']) + 4)

print('Done.')
