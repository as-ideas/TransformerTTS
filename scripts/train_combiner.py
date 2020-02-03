import os
import argparse

import ruamel.yaml
import tensorflow as tf
import numpy as np

from model.combiner import Combiner
from utils.preprocessing.data_handling import load_files
from utils.preprocessing.preprocessor import Preprocessor
from utils.train.logging import SummaryManager

np.random.seed(42)
tf.random.set_seed(42)


def linear_dropout_schedule(step):
    mx = config['decoder_prenet_dropout_schedule_max']
    mn = config['decoder_prenet_dropout_schedule_min']
    max_steps = config['decoder_prenet_dropout_schedule_max_steps']
    dout = max(((-mx + mn) / max_steps) * step + mx, mn)
    return tf.cast(dout, tf.float32)


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


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', dest='datadir', type=str)
parser.add_argument('--logdir', dest='log_dir', default='/tmp/summaries', type=str)
parser.add_argument('--config', dest='config', default='config/standard_config_0.yaml', type=str)
args = parser.parse_args()
yaml = ruamel.yaml.YAML()
config = yaml.load(open(args.config, 'r'))
config_name = os.path.splitext(os.path.basename(args.config))[0]
weights_paths, log_dir, base_dir = create_dirs(args, config)

print('creating model')
combiner = Combiner(config=config)

print('preprocessing data')
meldir = os.path.join(args.datadir, 'mels')
train_meta = os.path.join(args.datadir, 'train_metafile.txt')
test_meta = os.path.join(args.datadir, 'test_metafile.txt')

train_samples, _ = load_files(metafile=train_meta,
                              meldir=meldir,
                              num_samples=config['n_samples'])
test_samples, _ = load_files(metafile=test_meta,
                             meldir=meldir,
                             num_samples=config['n_samples'])

preprocessor = Preprocessor(mel_channels=config['mel_channels'],
                            start_vec_val=config['mel_start_vec_value'],
                            end_vec_val=config['mel_end_vec_value'],
                            tokenizer=combiner.tokenizer)
yaml.dump(config, open(os.path.join(base_dir, os.path.basename(args.config)), 'w'))
train_gen = lambda: (preprocessor(s) for s in train_samples)
test_list = [preprocessor(s) for s in test_samples]
train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float32, tf.int32, tf.int32))
train_dataset = train_dataset.shuffle(1000).padded_batch(config['batch_size'],
                                                         padded_shapes=([-1, 80], [-1], [-1]),
                                                         drop_remainder=True)

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
                                                max_to_keep=config['keep_n_weights'])
    # RESTORE LATEST MODEL
    checkpoints[kind].restore(managers[kind].latest_checkpoint)
    if managers[kind].latest_checkpoint:
        print(f'Restored {kind} from {managers[kind].latest_checkpoint}')
    else:
        print(f'Initializing {kind} from scratch.')

print('starting training')
decoder_prenet_dropout = config['fixed_decoder_prenet_dropout']
for epoch in range(config['epochs']):
    print(f'Epoch {epoch}')
    for (batch, (mel, text, stop)) in enumerate(train_dataset):
        if config['use_decoder_prenet_dropout_schedule']:
            decoder_prenet_dropout = linear_dropout_schedule(combiner.step)
        output = combiner.train_step(text=text,
                                     mel=mel,
                                     stop=stop,
                                     pre_dropout=decoder_prenet_dropout,
                                     mask_prob=config['mask_prob'])
        print(f'\nbatch {combiner.step}')
        
        summary_manager.write_loss(output, combiner.step)
        summary_manager.write_meta(name='dropout',
                                   value=decoder_prenet_dropout,
                                   step=combiner.step)
        summary_manager.write_meta(name='learning_rate',
                                   value=config['learning_rate'],
                                   step=combiner.step)
        
        for kind in transformer_kinds:
            losses[kind].append(float(output[kind]['loss']))
            
            if (combiner.step+1) % config['plot_attention_freq'] == 0:
                summary_manager.write_attention(output, combiner.step)
            print(f'{kind} mean loss: {sum(losses[kind]) / len(losses[kind])}')
            
            if (combiner.step+1) % config['weights_save_freq'] == 0:
                save_path = managers[kind].save()
                print(f'Saved checkpoint for step {combiner.step}: {save_path}')
                print(f'Saved checkpoint for step {combiner.step +1}: {save_path}')
        
        if (combiner.step +1) % config['image_freq'] == 0:
            for i in range(2):
                mel, text_seq, stop = test_list[i]
                text = combiner.tokenizer.decode(text_seq)
                pred = combiner.predict(mel,
                                        text_seq,
                                        pre_dropout=0.5,
                                        max_len_mel=mel.shape[0] + 50,
                                        max_len_text=len(text_seq) + 5)
                summary_manager.write_images(mel=mel, pred=pred, step=combiner.step, id=i)
                summary_manager.write_text(text=text, pred=pred, step=combiner.step)
