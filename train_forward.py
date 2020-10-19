from time import time

import tensorflow as tf
import numpy as np
from tqdm import trange

from utils.config_manager import Config
from preprocessing.datasets import TextMelDurDataset, ForwardPreprocessor
from utils.decorators import ignore_exception, time_it
from utils.scheduling import piecewise_linear_schedule, reduction_schedule
from utils.logging_utils import SummaryManager
from model.transformer_utils import create_mel_padding_mask
from utils.scripts_utils import dynamic_memory_allocation, basic_train_parser

np.random.seed(42)
tf.random.set_seed(42)
dynamic_memory_allocation()


@ignore_exception
@time_it
def validate(model,
             val_dataset,
             summary_manager):
    val_loss = {'loss': 0.}
    norm = 0.
    for mel, phonemes, durations, fname in val_dataset.all_batches():
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


parser = basic_train_parser()
args = parser.parse_args()

config = Config(config_path=args.config, model_kind='forward')
config_dict = config.config
config.create_remove_dirs(clear_dir=args.clear_dir,
                          clear_logs=args.clear_logs,
                          clear_weights=args.clear_weights)
config.dump_config()
config.print_config()

model = config.get_model()
config.compile_model(model)

data_prep = ForwardPreprocessor.from_config(config=config,
                                            tokenizer=model.text_pipeline.tokenizer)
train_data_handler = TextMelDurDataset.from_config(config,
                                                   preprocessor=data_prep,
                                                   kind='train')
valid_data_handler = TextMelDurDataset.from_config(config,
                                                   preprocessor=data_prep,
                                                   kind='valid')
train_dataset = train_data_handler.get_dataset(bucket_batch_sizes=config_dict['bucket_batch_sizes'],
                                               bucket_boundaries=config_dict['bucket_boundaries'],
                                               shuffle=True)
valid_dataset = valid_data_handler.get_dataset(bucket_batch_sizes=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1],
                                               bucket_boundaries=config_dict['bucket_boundaries'],
                                               shuffle=False,
                                               drop_remainder=True)

# create logger and checkpointer and restore latest model
summary_manager = SummaryManager(model=model, log_dir=config.log_dir, config=config_dict)
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 optimizer=model.optimizer,
                                 net=model)
manager = tf.train.CheckpointManager(checkpoint, config.weights_dir,
                                     max_to_keep=config_dict['keep_n_weights'],
                                     keep_checkpoint_every_n_hours=config_dict['keep_checkpoint_every_n_hours'])
manager_training = tf.train.CheckpointManager(checkpoint, str(config.weights_dir / 'latest'),
                                              max_to_keep=1, checkpoint_name='latest')

checkpoint.restore(manager_training.latest_checkpoint)
if manager_training.latest_checkpoint:
    print(f'\nresuming training from step {model.step} ({manager_training.latest_checkpoint})')
else:
    print(f'\nstarting training from scratch')

if config_dict['debug'] is True:
    print('\nWARNING: DEBUG is set to True. Training in eager mode.')
# main event
print('\nTRAINING')
losses = []
test_mel, test_phonemes, test_durs, test_fname = valid_dataset.next_batch()
t = trange(model.step, config_dict['max_steps'], leave=True)
for _ in t:
    t.set_description(f'step {model.step}')
    mel, phonemes, durations, fname = train_dataset.next_batch()
    learning_rate = piecewise_linear_schedule(model.step, config_dict['learning_rate_schedule'])
    decoder_prenet_dropout = piecewise_linear_schedule(model.step, config_dict['decoder_prenet_dropout_schedule'])
    drop_n_heads = tf.cast(reduction_schedule(model.step, config_dict['head_drop_schedule']), tf.int32)
    model.set_constants(decoder_prenet_dropout=decoder_prenet_dropout,
                        learning_rate=learning_rate,
                        drop_n_heads=drop_n_heads)
    output = model.train_step(input_sequence=phonemes,
                              target_sequence=mel,
                              target_durations=durations)
    losses.append(float(output['loss']))
    
    t.display(f'step loss: {losses[-1]}', pos=1)
    for pos, n_steps in enumerate(config_dict['n_steps_avg_losses']):
        if len(losses) > n_steps:
            t.display(f'{n_steps}-steps average loss: {sum(losses[-n_steps:]) / n_steps}', pos=pos + 2)
    
    summary_manager.display_loss(output, tag='Train')
    summary_manager.display_scalar(tag='Meta/learning_rate', scalar_value=model.optimizer.lr)
    summary_manager.display_scalar(tag='Meta/decoder_prenet_dropout', scalar_value=model.decoder_prenet.rate)
    summary_manager.display_scalar(tag='Meta/drop_n_heads', scalar_value=model.drop_n_heads)
    if model.step % config_dict['train_images_plotting_frequency'] == 0:
        summary_manager.display_attention_heads(output, tag='TrainAttentionHeads')
        summary_manager.display_mel(mel=output['mel'][0], tag=f'Train/predicted_mel')
        summary_manager.display_mel(mel=mel[0], tag=f'Train/target_mel')
        summary_manager.add_histogram(tag=f'Train/Predicted durations', values=output['duration'])
        summary_manager.add_histogram(tag=f'Train/Target durations', values=durations)
        summary_manager.display_audio(tag=f'Train/prediction', mel=output['mel'][0])
        summary_manager.display_audio(tag=f'Train/target', mel=mel[0])
    
    if model.step % 1000 == 0:
        save_path = manager_training.save()
    if model.step % config_dict['weights_save_frequency'] == 0:
        save_path = manager.save()
        t.display(f'checkpoint at step {model.step}: {save_path}', pos=len(config_dict['n_steps_avg_losses']) + 2)
    
    if model.step % config_dict['validation_frequency'] == 0:
        t.display(f'Validating', pos=len(config_dict['n_steps_avg_losses']) + 3)
        val_loss, time_taken = validate(model=model,
                                        val_dataset=valid_dataset,
                                        summary_manager=summary_manager)
        t.display(f'validation loss at step {model.step}: {val_loss} (took {time_taken}s)',
                  pos=len(config_dict['n_steps_avg_losses']) + 3)
    
    if model.step % config_dict['prediction_frequency'] == 0 and (model.step >= config_dict['prediction_start_step']):
        t.display(f'Predicting', pos=len(config_dict['n_steps_avg_losses']) + 4)
        timed_pred = time_it(model.predict)
        model_out, time_taken = timed_pred(test_phonemes, encode=False)
        summary_manager.display_attention_heads(model_out, tag='TestAttentionHeads')
        summary_manager.add_histogram(tag=f'Test/Predicted durations', values=model_out['duration'])
        summary_manager.add_histogram(tag=f'Test/Target durations', values=test_durs)
        pred_lengths = tf.cast(tf.reduce_sum(1 - model_out['expanded_mask'], axis=-1), tf.int32)
        pred_lengths = tf.squeeze(pred_lengths)
        tar_lengths = tf.cast(tf.reduce_sum(1 - create_mel_padding_mask(test_mel), axis=-1), tf.int32)
        tar_lengths = tf.squeeze(tar_lengths)
        display_start = time()
        for j, pred_mel in enumerate(model_out['mel']):
            predval = pred_mel[:pred_lengths[j], :]
            tar_value = test_mel[j, :tar_lengths[j], :]
            summary_manager.display_mel(mel=predval, tag=f'Test/{test_fname[j].numpy().decode("utf-8")}/predicted')
            summary_manager.display_mel(mel=tar_value, tag=f'Test/{test_fname[j].numpy().decode("utf-8")}/target')
            if j < config_dict['n_predictions']:
                if model.step >= config_dict['audio_start_step'] and (
                        model.step % config_dict['audio_prediction_frequency'] == 0):
                    summary_manager.display_audio(tag=f'{test_fname[j].numpy().decode("utf-8")}/target', mel=tar_value)
                    summary_manager.display_audio(tag=f'{test_fname[j].numpy().decode("utf-8")}/prediction',
                                                  mel=predval)
            else:
                break
        display_end = time()
        t.display(f'Predictions took {time_taken}. Displaying took {display_end - display_start}.',
                  pos=len(config_dict['n_steps_avg_losses']) + 4)
print('Done.')
