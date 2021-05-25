import tensorflow as tf
import numpy as np
from tqdm import trange

from utils.training_config_manager import TrainingConfigManager
from data.datasets import TTSDataset, TTSPreprocessor
from utils.decorators import ignore_exception, time_it
from utils.scheduling import piecewise_linear_schedule
from utils.logging_utils import SummaryManager
from model.transformer_utils import create_mel_padding_mask
from utils.scripts_utils import dynamic_memory_allocation, basic_train_parser
from data.metadata_readers import post_processed_reader

np.random.seed(42)
tf.random.set_seed(42)
dynamic_memory_allocation()


def display_target_symbol_duration_distributions():
    phon_data, ups = post_processed_reader(config.phonemized_metadata_path)
    dur_dict = {}
    for key in phon_data.keys():
        dur_dict[key] = np.load((config.duration_dir / key).with_suffix('.npy'))
    symbol_durs = {}
    for key in dur_dict:
        for i, phoneme in enumerate(phon_data[key]):
            symbol_durs.setdefault(phoneme, []).append(dur_dict[key][i])
    for symbol in symbol_durs.keys():
        summary_manager.add_histogram(tag=f'"{symbol}"/Target durations', values=symbol_durs[symbol],
                                      buckets=len(set(symbol_durs[symbol])) + 1, step=0)


def display_predicted_symbol_duration_distributions(all_durations):
    phon_data, ups = post_processed_reader(config.phonemized_metadata_path)
    symbol_durs = {}
    for key in all_durations.keys():
        clean_key = key.decode('utf-8')
        for i, phoneme in enumerate(phon_data[clean_key]):
            symbol_durs.setdefault(phoneme, []).append(all_durations[key][i])
    for symbol in symbol_durs.keys():
        summary_manager.add_histogram(tag=f'"{symbol}"/Predicted durations', values=symbol_durs[symbol])


@ignore_exception
@time_it
def validate(model,
             val_dataset,
             summary_manager):
    val_loss = {'loss': 0.}
    norm = 0.
    for mel, phonemes, durations, pitch, fname in val_dataset.all_batches():
        model_out = model.val_step(input_sequence=phonemes,
                                   target_sequence=mel,
                                   target_durations=durations,
                                   target_pitch=pitch)
        norm += 1
        val_loss['loss'] += model_out['loss']
    val_loss['loss'] /= norm
    summary_manager.display_loss(model_out, tag='Validation', plot_all=True)
    summary_manager.display_attention_heads(model_out, tag='ValidationAttentionHeads')
    summary_manager.add_histogram(tag=f'Validation/Predicted durations', values=model_out['duration'])
    summary_manager.add_histogram(tag=f'Validation/Target durations', values=durations)
    summary_manager.display_plot1D(tag=f'Validation/{fname[0].numpy().decode("utf-8")} predicted pitch',
                                   y=model_out['pitch'][0])
    summary_manager.display_plot1D(tag=f'Validation/{fname[0].numpy().decode("utf-8")} target pitch', y=pitch[0])
    summary_manager.display_mel(mel=model_out['mel'][0],
                                tag=f'Validation/{fname[0].numpy().decode("utf-8")} predicted_mel')
    summary_manager.display_mel(mel=mel[0], tag=f'Validation/{fname[0].numpy().decode("utf-8")} target_mel')
    summary_manager.display_audio(tag=f'Validation {fname[0].numpy().decode("utf-8")}/prediction',
                                  mel=model_out['mel'][0])
    summary_manager.display_audio(tag=f'Validation {fname[0].numpy().decode("utf-8")}/target', mel=mel[0])
    # predict withoyt enforcing durations and pitch
    model_out = model.predict(phonemes, encode=False)
    pred_lengths = tf.cast(tf.reduce_sum(1 - model_out['expanded_mask'], axis=-1), tf.int32)
    pred_lengths = tf.squeeze(pred_lengths)
    tar_lengths = tf.cast(tf.reduce_sum(1 - create_mel_padding_mask(mel), axis=-1), tf.int32)
    tar_lengths = tf.squeeze(tar_lengths)
    for j, pred_mel in enumerate(model_out['mel']):
        predval = pred_mel[:pred_lengths[j], :]
        tar_value = mel[j, :tar_lengths[j], :]
        summary_manager.display_mel(mel=predval, tag=f'Test/{fname[j].numpy().decode("utf-8")}/predicted')
        summary_manager.display_mel(mel=tar_value, tag=f'Test/{fname[j].numpy().decode("utf-8")}/target')
        summary_manager.display_audio(tag=f'Prediction {fname[j].numpy().decode("utf-8")}/target', mel=tar_value)
        summary_manager.display_audio(tag=f'Prediction {fname[j].numpy().decode("utf-8")}/prediction',
                                      mel=predval)
    return val_loss['loss']


parser = basic_train_parser()
args = parser.parse_args()

config = TrainingConfigManager(config_path=args.config)
config_dict = config.config
config.create_remove_dirs(clear_dir=args.clear_dir,
                          clear_logs=args.clear_logs,
                          clear_weights=args.clear_weights)
config.dump_config()
config.print_config()

model = config.get_model()
config.compile_model(model)

data_prep = TTSPreprocessor.from_config(config=config,
                                        tokenizer=model.text_pipeline.tokenizer)
train_data_handler = TTSDataset.from_config(config,
                                            preprocessor=data_prep,
                                            kind='train')
valid_data_handler = TTSDataset.from_config(config,
                                            preprocessor=data_prep,
                                            kind='valid')
train_dataset = train_data_handler.get_dataset(bucket_batch_sizes=config_dict['bucket_batch_sizes'],
                                               bucket_boundaries=config_dict['bucket_boundaries'],
                                               shuffle=True)
valid_dataset = valid_data_handler.get_dataset(bucket_batch_sizes=config_dict['val_bucket_batch_size'],
                                               bucket_boundaries=config_dict['bucket_boundaries'],
                                               shuffle=False,
                                               drop_remainder=True)

# create logger and checkpointer and restore latest model
summary_manager = SummaryManager(model=model, log_dir=config.log_dir, config=config_dict)
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 optimizer=model.optimizer,
                                 net=model)
manager_training = tf.train.CheckpointManager(checkpoint, str(config.weights_dir / 'latest'),
                                              max_to_keep=1, checkpoint_name='latest')

checkpoint.restore(manager_training.latest_checkpoint)
if manager_training.latest_checkpoint:
    print(f'\nresuming training from step {model.step} ({manager_training.latest_checkpoint})')
else:
    print(f'\nstarting training from scratch')

if config_dict['debug'] is True:
    print('\nWARNING: DEBUG is set to True. Training in eager mode.')

display_target_symbol_duration_distributions()
# main event
print('\nTRAINING')
losses = []
texts = []
for text_file in config_dict['text_prediction']:
    with open(text_file, 'r') as file:
        text = file.readlines()
    texts.append(text)

all_files = len(set(train_data_handler.metadata_reader.filenames))  # without duplicates
all_durations = {}
t = trange(model.step, config_dict['max_steps'], leave=True)
for _ in t:
    t.set_description(f'step {model.step}')
    mel, phonemes, durations, pitch, fname = train_dataset.next_batch()
    learning_rate = piecewise_linear_schedule(model.step, config_dict['learning_rate_schedule'])
    model.set_constants(learning_rate=learning_rate)
    output = model.train_step(input_sequence=phonemes,
                              target_sequence=mel,
                              target_durations=durations,
                              target_pitch=pitch)
    losses.append(float(output['loss']))
    
    predicted_durations = dict(zip(fname.numpy(), output['duration'].numpy()))
    all_durations.update(predicted_durations)
    if len(all_durations) >= all_files:  # all the dataset has been processed
        display_predicted_symbol_duration_distributions(all_durations)
        all_durations = {}
    
    t.display(f'step loss: {losses[-1]}', pos=1)
    for pos, n_steps in enumerate(config_dict['n_steps_avg_losses']):
        if len(losses) > n_steps:
            t.display(f'{n_steps}-steps average loss: {sum(losses[-n_steps:]) / n_steps}', pos=pos + 2)
    
    summary_manager.display_loss(output, tag='Train')
    summary_manager.display_scalar(scalar_value=t.avg_time, tag='Meta/iter_time')
    summary_manager.display_scalar(scalar_value=tf.shape(fname)[0], tag='Meta/batch_size')
    summary_manager.display_scalar(tag='Meta/learning_rate', scalar_value=model.optimizer.lr)
    if model.step % config_dict['train_images_plotting_frequency'] == 0:
        summary_manager.display_attention_heads(output, tag='TrainAttentionHeads')
        summary_manager.display_mel(mel=output['mel'][0], tag=f'Train/predicted_mel')
        summary_manager.display_mel(mel=mel[0], tag=f'Train/target_mel')
        summary_manager.display_plot1D(tag=f'Train/Predicted pitch', y=output['pitch'][0])
        summary_manager.display_plot1D(tag=f'Train/Target pitch', y=pitch[0])
    
    if model.step % 1000 == 0:
        save_path = manager_training.save()
    if (model.step % config_dict['weights_save_frequency'] == 0) & (
            model.step >= config_dict['weights_save_starting_step']):
        model.save_model(config.weights_dir / f'step_{model.step}')
        t.display(f'checkpoint at step {model.step}: {config.weights_dir / f"step_{model.step}"}',
                  pos=len(config_dict['n_steps_avg_losses']) + 2)
    
    if model.step % config_dict['validation_frequency'] == 0:
        t.display(f'Validating', pos=len(config_dict['n_steps_avg_losses']) + 3)
        val_loss, time_taken = validate(model=model,
                                        val_dataset=valid_dataset,
                                        summary_manager=summary_manager)
        t.display(f'validation loss at step {model.step}: {val_loss} (took {time_taken}s)',
                  pos=len(config_dict['n_steps_avg_losses']) + 3)
    
    if model.step % config_dict['prediction_frequency'] == 0 and (model.step >= config_dict['prediction_start_step']):
        for i, text in enumerate(texts):
            wavs = []
            for i, text_line in enumerate(text):
                out = model.predict(text_line, encode=True)
                wav = summary_manager.audio.reconstruct_waveform(out['mel'].numpy().T)
                wavs.append(wav)
            wavs = np.concatenate(wavs)
            wavs = tf.expand_dims(wavs, 0)
            wavs = tf.expand_dims(wavs, -1)
            summary_manager.add_audio(f'Text file input', wavs.numpy(), sr=summary_manager.config['sampling_rate'],
                                      step=summary_manager.global_step)

print('Done.')
