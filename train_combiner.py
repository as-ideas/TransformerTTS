import os
import argparse

import ruamel.yaml
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from model.combiner import Combiner
from preprocessing.utils import load_files
from preprocessing.preprocessor import Preprocessor
from utils import plot_attention, display_mel

np.random.seed(42)
tf.random.set_seed(42)


def linear_dropout_schedule(step):
    mx = config['decoder_prenet_dropout_schedule_max']
    mn = config['decoder_prenet_dropout_schedule_min']
    max_steps = config['decoder_prenet_dropout_schedule_max_steps']
    dout = max(((-mx + mn) / max_steps) * step + mx, mn)
    return tf.cast(dout, tf.float32)


def create_dirs(args, config):
    args.log_dir = os.path.join(args.log_dir, config_name)
    weights_paths = os.path.join(args.log_dir, f'weights/')
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(weights_paths, exist_ok=True)
    weights_paths = {}
    for kind in config['transformer_kinds']:
        weights_paths[kind] = os.path.join(args.log_dir, f'weights/{kind}/')
    return weights_paths


class SummaryManager:

    def __init__(self,
                 log_dir,
                 transformer_kinds):
        self.all_kinds = transformer_kinds
        self.text_kinds = [k for k in transformer_kinds
                           if k in ['text_text', 'mel_text']]
        self.mel_kinds = [k for k in transformer_kinds
                          if k in ['mel_mel', 'text_mel']]
        self.summary_writers = {}
        for kind in transformer_kinds:
            path = os.path.join(log_dir, kind)
            self.summary_writers[kind] = tf.summary.create_file_writer(path)
        meta_path = os.path.join(log_dir, 'meta')
        self.summary_writers['meta'] = tf.summary.create_file_writer(meta_path)

    def write_images(self, mel, pred, step):
        for kind in self.mel_kinds:
            self._write_image(kind,
                              mel=mel,
                              pred=pred[kind],
                              step=step)

    def write_text(self, text, pred, step):
        for kind in self.text_kinds:
            pred_decoded = pred[kind]['output_decoded']
            self._write_text(kind, text, pred_decoded, step)

    def write_loss(self, output, step):
        for kind in self.all_kinds:
            with self.summary_writers[kind].as_default():
                loss = output[kind]['loss']
                tf.summary.scalar('loss', loss, step=step)
                if kind in self.mel_kinds:
                    for k in output[kind]['losses'].keys():
                        loss = output[kind]['losses'][k]
                        tf.summary.scalar(kind + '_' + k, loss, step=step)

    def write_meta(self, name, value, step):
        with self.summary_writers['meta'].as_default():
            tf.summary.scalar(name, tf.Variable(value), step=step)

    def write_attention(self, output, step):
        for kind in self.all_kinds:
            with self.summary_writers[kind].as_default():
                plot_attention(outputs=output[kind],
                               step=step,
                               info_string='train attention ')

    def _write_image(self, kind, mel, pred, step):
        with self.summary_writers[kind].as_default():
            plot_attention(outputs=pred,
                           step=step,
                           info_string='test attention ')
            display_mel(mel=pred['mel'],
                        step=step,
                        info_string='test mel {}'.format(i))
            display_mel(mel=mel,
                        step=step,
                        info_string='target mel {}'.format(i))

    def _write_text(self, kind, text, pred_decoded, step):
        with self.summary_writers[kind].as_default():
            name = u'{} from validation'.format(kind)
            data_pred = u'(pred) {}'.format(pred_decoded)
            data_target = u'(target) {}'.format(text)
            tf.summary.text(name=name, data=data_pred, step=step)
            tf.summary.text(name=name, data=data_target, step=step)


parser = argparse.ArgumentParser()
parser.add_argument('--meldir', dest='meldir', type=str)
parser.add_argument('--metafile', dest='metafile', type=str)
parser.add_argument('--logdir', dest='log_dir', type=str)
parser.add_argument('--config', dest='config', type=str)
args = parser.parse_args()
yaml = ruamel.yaml.YAML()
config = yaml.load(open(args.config, 'r'))
config_name = os.path.splitext(os.path.basename(args.config))[0]
weights_paths = create_dirs(args, config)

print('creating model')
combiner = Combiner(config=config)

print('preprocessing data')
samples, alphabet = load_files(metafile=args.metafile,
                               meldir=args.meldir,
                               num_samples=config['n_samples'])
train_samples, test_samples = train_test_split(samples, test_size=100, random_state=42)
preprocessor = Preprocessor(mel_channels=config['mel_channels'],
                            start_vec_val=config['mel_start_vec_value'],
                            end_vec_val=config['mel_end_vec_value'],
                            tokenizer=combiner.tokenizer)
yaml.dump(config, open(os.path.join(args.log_dir, os.path.basename(args.config)), 'w'))
train_gen = lambda: (preprocessor(s) for s in train_samples)
test_list = [preprocessor(s) for s in test_samples]
train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float32, tf.int64, tf.int64))
train_dataset = train_dataset.shuffle(1000).padded_batch(config['batch_size'],
                                                         padded_shapes=([-1, 80], [-1], [-1]),
                                                         drop_remainder=True)

losses = {}
summary_writers = {}
checkpoints = {}
managers = {}
transformer_kinds = config['transformer_kinds']
summary_manager = SummaryManager(args.log_dir, transformer_kinds)

for kind in transformer_kinds:
    path = os.path.join(args.log_dir, kind)
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
            decoder_prenet_dropout = linear_dropout_schedule(int(checkpoints[transformer_kinds[0]].step))
        output = combiner.train_step(text=text,
                                     mel=mel,
                                     stop=stop,
                                     pre_dropout=decoder_prenet_dropout,
                                     mask_prob=config['mask_prob'])
        print(f'\nbatch {int(combiner.step)}')

        summary_manager.write_loss(output, combiner.step)
        summary_manager.write_meta(name='dropout',
                                   value=decoder_prenet_dropout,
                                   step=combiner.step)
        summary_manager.write_meta(name='learning_rate',
                                   value=config['learning_rate'],
                                   step=combiner.step)

        for kind in transformer_kinds:
            checkpoints[kind].step.assign_add(1)
            losses[kind].append(float(output[kind]['loss']))

            if int(checkpoints[kind].step) % config['plot_attention_freq'] == 0:
                summary_manager.write_attention(output, combiner.step)
            print(f'{kind} mean loss: {sum(losses[kind]) / len(losses[kind])}')

            if int(checkpoints[kind].step) % config['weights_save_freq'] == 0:
                save_path = managers[kind].save()
                print(f'Saved checkpoint for step {int(checkpoints[kind].step)}: {save_path}')

        if int(checkpoints[transformer_kinds[0]].step) % config['image_freq'] == 0:
            for i in range(2):
                mel, text_seq, stop = test_list[i]
                text = combiner.tokenizer.decode(text_seq)
                pred = combiner.predict(mel,
                                        text_seq,
                                        pre_dropout=0.5,
                                        max_len_mel=mel.shape[0] + 50,
                                        max_len_text=len(text_seq) + 5)
                summary_manager.write_images(mel=mel, pred=pred, step=combiner.step)
                summary_manager.write_text(text=text, pred=pred, step=combiner.step)

