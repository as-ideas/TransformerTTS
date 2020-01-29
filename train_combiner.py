import os
import argparse

import ruamel.yaml
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from model.transformer_factory import Combiner
from preprocessing.utils import load_files
from preprocessing.preprocessor import Preprocessor
from utils import plot_attention, display_mel

np.random.seed(42)
tf.random.set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--meldir', dest='meldir', type=str)
parser.add_argument('--metafile', dest='metafile', type=str)
parser.add_argument('--logdir', dest='log_dir', type=str)
parser.add_argument('--config', dest='config', type=str)
args = parser.parse_args()
yaml = ruamel.yaml.YAML()
config = yaml.load(open(args.config, 'r'))
config_name = os.path.splitext(os.path.basename(args.config))[0]


def linear_dropout_schedule(step):
    mx = config['decoder_prenet_dropout_schedule_max']
    mn = config['decoder_prenet_dropout_schedule_min']
    max_steps = config['decoder_prenet_dropout_schedule_max_steps']
    dout = max(((-mx + mn) / max_steps) * step + mx, mn)
    return tf.cast(dout, tf.float32)


# CREATE DIRS AND PATHS
args.log_dir = os.path.join(args.log_dir, config_name)
weights_paths = os.path.join(args.log_dir, f'weights/')
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(weights_paths, exist_ok=True)
weights_paths = {}
for kind in config['transformer_kinds']:
    weights_paths[kind] = os.path.join(args.log_dir, f'weights/{kind}/')
    
samples, alphabet = load_files(metafile=args.metafile,
                               meldir=args.meldir,
                               num_samples=config['n_samples'])

print('creating model')
combiner = Combiner(config=config, tokenizer_alphabet=alphabet)
print('preprocessing data')
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
# PREPARE LOGGING
losses = {}
summary_writers = {}
checkpoints = {}
managers = {}
transformer_kinds = config['transformer_kinds']
for kind in transformer_kinds:
    path = os.path.join(args.log_dir, kind)
    summary_writers[kind] = tf.summary.create_file_writer(path)
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

decoder_prenet_dropout = config['fixed_decoder_prenet_dropout']
print('Starting training')
for epoch in range(config['epochs']):
    print(f'Epoch {epoch}')
    for (batch, (mel, text, stop)) in enumerate(train_dataset):
        if config['use_decoder_prenet_dropout_schedule']:
            decoder_prenet_dropout = linear_dropout_schedule(int(checkpoints[transformer_kinds[0]].step))
        # set_learning_rate(int(checkpoints[transformer_kinds[0]].step))
        output = combiner.train_step(text=text,
                                     mel=mel,
                                     stop=stop,
                                     speech_decoder_prenet_dropout=decoder_prenet_dropout,
                                     mask_prob=config['mask_prob'],
                                     )
        print(f'\nbatch {int(checkpoints[config["transformer_kinds"][0]].step)}')
        
        # CHECKPOINTING
        first_kind = transformer_kinds[0]
        step = getattr(combiner, first_kind).optimizer.iterations
        for kind in transformer_kinds:
            checkpoints[kind].step.assign_add(1)
            losses[kind].append(float(output[kind]['loss']))
            with summary_writers[kind].as_default():
                if (kind == 'text_mel') or (kind == 'mel_mel'):
                    for k in output[kind]['losses'].keys():
                        tf.summary.scalar(kind + '_' + k, output[kind]['losses'][k], step=step)
                tf.summary.scalar('loss', output[kind]['loss'], step=step)
            print(f'{kind} mean loss: {sum(losses[kind]) / len(losses[kind])}')

            if int(checkpoints[kind].step) % config['weights_save_freq'] == 0:
                save_path = managers[kind].save()
                print(f'Saved checkpoint for step {int(checkpoints[kind].step)}: {save_path}')
            
            if int(checkpoints[kind].step) % config['plot_attention_freq'] == 0:
                with summary_writers[kind].as_default():
                    plot_attention(output[kind], step=step, info_string='train attention ')
        
        with summary_writers[transformer_kinds[0]].as_default():
            first_kind = transformer_kinds[0]
            tf.summary.scalar('dropout', decoder_prenet_dropout, step=step)
            tf.summary.scalar(name='learning_rate',
                              data=getattr(combiner, first_kind).optimizer.lr,
                              step=step)
            
        # PREDICT MEL
        if ('mel_mel' in transformer_kinds) and ('text_mel' in transformer_kinds):
            if int(checkpoints[transformer_kinds[0]].step) % config['image_freq'] == 0:
                pred = {}
                test_val = {}
                for i in range(0, 2):
                    mel_target = test_list[i][0]
                    max_pred_len = mel_target.shape[0] + 50
                    test_val['text_mel'] = test_list[i][1]
                    test_val['mel_mel'] = mel_target
                    for kind in ['text_mel', 'mel_mel']:
                        pred[kind] = getattr(combiner, kind).predict(test_val[kind],
                                                                     max_length=max_pred_len,
                                                                     decoder_prenet_dropout=0.5)
                        with summary_writers[kind].as_default():
                            plot_attention(outputs=pred[kind],
                                           step=step,
                                           info_string='test attention ')
                            display_mel(pred=pred[kind]['mel'],
                                        step=step,
                                        info_string='test mel {}'.format(i))
                            display_mel(pred=mel_target,
                                        step=step,
                                        info_string='target mel {}'.format(i))
        # PREDICT TEXT
        if ('mel_text' in transformer_kinds) and ('text_text' in transformer_kinds):
            if int(checkpoints[transformer_kinds[0]].step) % config['text_freq'] == 0:
                pred = {}
                test_val = {}
                for i in range(0, 2):
                    test_val['mel_text'] = test_list[i][0]
                    test_val['text_text'] = test_list[i][1]
                    decoded_target = combiner.tokenizer.decode(test_val['text_text'])
                    for kind in ['mel_text', 'text_text']:
                        pred[kind] = getattr(combiner, kind).predict(test_val[kind])
                        pred[kind] = combiner.tokenizer.decode(pred[kind]['output'])
                        with summary_writers[kind].as_default():
                            tf.summary.text(name=f'{kind} from validation',
                                            data=f'(pred) {pred[kind]}\n(target) {decoded_target}',
                                            step=step)
