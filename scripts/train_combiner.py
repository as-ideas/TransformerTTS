import os
import argparse

import ruamel.yaml
import tensorflow as tf
import numpy as np

from model.combiner import Combiner
from preprocessing.data_handling import load_files, Dataset
from preprocessing.preprocessor import Preprocessor
from utils.decorators import ignore_exception, time_it
from utils.scheduling import dropout_schedule, learning_rate_schedule
from utils.logging import SummaryManager

np.random.seed(42)
tf.random.set_seed(42)


def create_dirs(args, config):
    base_dir = os.path.join(args.log_dir, config_name)
    log_dir = os.path.join(base_dir, f'logs/')
    weights_dir = os.path.join(base_dir, f'weights/')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    weights_paths = {}
    for kind in config['transformer_kinds']:
        weights_paths[kind] = os.path.join(weights_dir, kind)
    return weights_paths, log_dir, base_dir


@ignore_exception
@time_it
def validate(combiner,
             val_dataset,
             summary_manager,
             decoder_prenet_dropout):
    print(f'\nvalidating at step {combiner.step}')
    val_loss = {kind: {'loss': 0.} for kind in combiner.transformer_kinds}
    norm = 0.
    for val_mel, val_text, val_stop in val_dataset.all_batches():
        model_out = combiner.val_step(val_text,
                                      val_mel,
                                      val_stop,
                                      pre_dropout=decoder_prenet_dropout,
                                      mask_prob=0.)
        norm += 1
        for kind in model_out.keys():
            val_loss[kind]['loss'] += model_out[kind]['loss']
    for kind in val_loss.keys():
        val_loss[kind]['loss'] /= norm
        val_loss_kind = val_loss[kind]['loss']
        print(f'{kind} val loss: {val_loss_kind}')
    summary_manager.write_loss(val_loss, combiner.step, name='val_loss')


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', dest='datadir', type=str)
parser.add_argument('--logdir', dest='log_dir', default='/tmp/summaries', type=str)
parser.add_argument('--config', dest='config', type=str)
args = parser.parse_args()
yaml = ruamel.yaml.YAML()
config = yaml.load(open(args.config, 'r'))
config_name = os.path.splitext(os.path.basename(args.config))[0]
weights_paths, log_dir, base_dir = create_dirs(args, config)

combiner = Combiner(config=config)
meldir = os.path.join(args.datadir, 'mels')
train_meta = os.path.join(args.datadir, 'train_metafile.txt')
test_meta = os.path.join(args.datadir, 'test_metafile.txt')

train_samples, _ = load_files(metafile=train_meta,
                              meldir=meldir,
                              num_samples=config['n_samples'])
val_samples, _ = load_files(metafile=test_meta,
                            meldir=meldir,
                            num_samples=config['n_samples'])

preprocessor = Preprocessor(mel_channels=config['mel_channels'],
                            start_vec_val=config['mel_start_vec_value'],
                            end_vec_val=config['mel_end_vec_value'],
                            tokenizer=combiner.tokenizer)

yaml.dump(config, open(os.path.join(base_dir, os.path.basename(args.config)), 'w'))
test_list = [preprocessor(s) for s in val_samples]

train_dataset = Dataset(samples=train_samples,
                        preprocessor=preprocessor,
                        batch_size=config['batch_size'],
                        shuffle=True)
val_dataset = Dataset(samples=val_samples,
                      preprocessor=preprocessor,
                      batch_size=config['batch_size'],
                      shuffle=False)

val_list = [preprocessor(s) for s in val_samples]

losses = {}
summary_writers = {}
checkpoints = {}
managers = {}
transformer_kinds = config['transformer_kinds']
summary_manager = SummaryManager(log_dir, transformer_kinds)

for kind in transformer_kinds:
    path = os.path.join(log_dir, kind)
    losses[kind] = []
    checkpoints[kind] = tf.train.Checkpoint(step=tf.Variable(1),
                                            optimizer=getattr(combiner, kind).optimizer,
                                            net=getattr(combiner, kind))
    managers[kind] = tf.train.CheckpointManager(checkpoints[kind], weights_paths[kind],
                                                max_to_keep=config['keep_n_weights'],
                                                keep_checkpoint_every_n_hours=config['keep_checkpoint_every_n_hours'])
    # restore latest model
    checkpoints[kind].restore(managers[kind].latest_checkpoint)
    if managers[kind].latest_checkpoint:
        print(f'restored {kind} from {managers[kind].latest_checkpoint}')
    else:
        print(f'initializing {kind} from scratch')

print('starting training')
while combiner.step < config['max_steps']:
    mel, text, stop = train_dataset.next_batch()
    decoder_prenet_dropout = dropout_schedule(combiner.step, config['dropout_schedule'])
    learning_rate = learning_rate_schedule(combiner.step, config['learning_rate_schedule'])
    combiner.set_learning_rates(learning_rate)

    output = combiner.train_step(text=text,
                                 mel=mel,
                                 stop=stop,
                                 pre_dropout=decoder_prenet_dropout,
                                 mask_prob=config['mask_prob'])
    print(f'\nbatch {combiner.step}')

    summary_manager.write_loss(output, combiner.step)
    summary_manager.write_meta_scalar(name='dropout',
                                      value=decoder_prenet_dropout,
                                      step=combiner.step)

    for kind in transformer_kinds:
        losses[kind].append(float(output[kind]['loss']))
        summary_manager.write_meta_for_kind(name='learning_rate',
                                            value=getattr(combiner, kind).optimizer.lr,
                                            step=combiner.step,
                                            kind=kind)

        if (combiner.step + 1) % config['plot_attention_freq'] == 0:
            summary_manager.write_attention(output, combiner.step)
        print(f'{kind} mean loss: {sum(losses[kind]) / len(losses[kind])}')

        if (combiner.step + 1) % config['weights_save_freq'] == 0:
            save_path = managers[kind].save()
            print(f'Saved checkpoint for step {combiner.step}: {save_path}')

    if (combiner.step + 1) % config['val_freq'] == 0:
        validate(combiner=combiner,
                 val_dataset=val_dataset,
                 summary_manager=summary_manager,
                 decoder_prenet_dropout=decoder_prenet_dropout)

    if (combiner.step + 1) % config['text_freq'] == 0:
        for i in range(2):
            mel, text_seq, stop = test_list[i]
            text = combiner.tokenizer.decode(text_seq)
            pred = combiner.predict(mel,
                                    text_seq,
                                    pre_dropout=decoder_prenet_dropout,
                                    max_len_text=len(text_seq) + 5,
                                    max_len_mel=False)
            summary_manager.write_text(text=text, pred=pred, step=combiner.step)

    if (combiner.step + 1) % config['image_freq'] == 0:
        for i in range(2):
            mel, text_seq, stop = test_list[i]
            text = combiner.tokenizer.decode(text_seq)
            pred = combiner.predict(mel,
                                    text_seq,
                                    pre_dropout=decoder_prenet_dropout,
                                    max_len_mel=mel.shape[0] + 50,
                                    max_len_text=False)
            summary_manager.write_images(mel=mel,
                                         pred=pred,
                                         step=combiner.step,
                                         id=i)
            summary_manager.write_audios(mel=mel,
                                         pred=pred,
                                         config=config,
                                         step=combiner.step,
                                         id=i)

    if combiner.step >= config['max_steps']:
        print(f'Stopping training at step {combiner.step}.')
        break

print('done fucker.')
