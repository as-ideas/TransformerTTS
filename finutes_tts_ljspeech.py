import tensorflow as tf
import numpy as np
from tqdm import trange

from utils.training_config_manager import TrainingConfigManager
from data.datasets import TTSDataset, TTSPreprocessor, DataReader
from utils.scheduling import piecewise_linear_schedule
from utils.logging_utils import SummaryManager
from utils.scripts_utils import dynamic_memory_allocation, basic_train_parser
from model.models import ForwardTransformer

np.random.seed(42)
tf.random.set_seed(42)
dynamic_memory_allocation()


def display_target_symbol_duration_distributions():
    dur_dict = {}
    for key in phonemized_metadata.filenames:
        dur_dict[key] = np.load((config.duration_dir / key).with_suffix('.npy'))
    symbol_durs = {}
    for key in dur_dict:
        for i, phoneme in enumerate(phonemized_metadata.text_dict[key]):
            symbol_durs.setdefault(phoneme, []).append(dur_dict[key][i])
    for symbol in symbol_durs.keys():
        summary_manager.add_histogram(tag=f'"{symbol}"/Target durations', values=symbol_durs[symbol],
                                      buckets=len(set(symbol_durs[symbol])) + 1, step=0)


def display_predicted_symbol_duration_distributions(all_durations):
    symbol_durs = {}
    for key in all_durations.keys():
        clean_key = key.decode('utf-8')
        for i, phoneme in enumerate(phonemized_metadata.text_dict[clean_key]):
            symbol_durs.setdefault(phoneme, []).append(all_durations[key][i])
    for symbol in symbol_durs.keys():
        summary_manager.add_histogram(tag=f'"{symbol}"/Predicted durations', values=symbol_durs[symbol])


parser = basic_train_parser()
parser.add_argument('-m', '--model_path', dest='model_path', type=str)
args = parser.parse_args()
config = TrainingConfigManager.from_config(config_path=args.config)
new_config = config.config
new_config['data_name'] = new_config['cloned_voice_name']
config = TrainingConfigManager(config=new_config, model_kind='tts')
config_dict = config.config
config.create_remove_dirs(clear_dir=args.clear_dir,
                          clear_logs=args.clear_logs,
                          clear_weights=args.clear_weights)
config.dump_config()
config.print_config()

# model = config.get_model()
model = ForwardTransformer.load_model(args.model_path)
print(f"Loaded model from step {model.config['step']}")
config.compile_model(model)

data_prep = TTSPreprocessor.from_config(config=config,
                                        tokenizer=model.text_pipeline.tokenizer)

train_data_reader = DataReader.from_config(config, kind='train')
train_data_reader.filenames = list(set(train_data_reader.filenames))[:config_dict['finetuning_n_samples']]
train_data_handler = TTSDataset.from_config(config,
                                            preprocessor=data_prep,
                                            kind='train',
                                            metadata_reader=train_data_reader)
train_dataset = train_data_handler.get_dataset(bucket_batch_sizes=config_dict['bucket_batch_sizes'],
                                               bucket_boundaries=config_dict['bucket_boundaries'],
                                               shuffle=True)

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

phonemized_metadata = DataReader.from_config(config, kind='phonemized', scan_wavs=False)
display_target_symbol_duration_distributions()

# check amount of finetuning data
tot_lens = 0
for mel, phonemes, durations, pitch, fname, reference_wav_embedding, speaker_id in train_dataset.all_batches():
    durs = tf.reduce_sum(durations)
    tot_lens += durs
summary_manager.display_scalar('Total duration (hours)',
                               scalar_value=(tot_lens * config_dict['hop_length']) / config_dict[
                                   'sampling_rate'] / 60. ** 2)
summary_manager.display_scalar('Total duration (mels)',
                               scalar_value=tot_lens)

# main event
print('\nTRAINING')
losses = []
with open(config_dict['text_prediction'], 'r') as file:
    text = file.readlines()
test_text_batch = [model.encode_text(text_line) for text_line in text]
pad = len(max(test_text_batch, key=len))
test_text_batch = np.array([np.pad(i, (0, pad - len(i)), constant_values=0) for i in test_text_batch])

all_files = len(set(train_data_handler.metadata_reader.filenames))  # without duplicates
all_durations = {}
t = trange(model.step, config_dict['max_steps'], leave=True)
for _ in t:
    t.set_description(f'step {model.step}')
    mel, phonemes, durations, pitch, fname, reference_wav_embedding, speaker_id = train_dataset.next_batch()
    learning_rate = piecewise_linear_schedule(model.step, config_dict['learning_rate_schedule'])
    model.set_constants(learning_rate=learning_rate)
    output = model.train_step(input_sequence=phonemes,
                              target_sequence=mel,
                              target_durations=durations,
                              target_pitch=pitch,
                              reference_wav_embedding=reference_wav_embedding)
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
    
    did_start = model.step >= config_dict['prediction_start_step']
    is_period = model.step % config_dict['prediction_frequency'] == 0
    if did_start and is_period:
        test_mels, _, _, _, _, test_reference_wav_embeddings, test_speaker_ids = train_dataset.next_batch()
        for ref_mel, ref_wav_emb, test_speaker_id in zip(test_mels, test_reference_wav_embeddings, test_speaker_ids):
            out = model.predict(test_text_batch,
                                ref_wav_emb[None],
                                encode=False)
            
            mel_mask = tf.reshape((1 - out['expanded_mask'][:, 0, 0, :]) != 0, -1)
            concat_mel = tf.reshape(out['mel'], [-1, 80])
            concat_mel = concat_mel.numpy()[mel_mask]
            test_wavs = summary_manager.audio.reconstruct_waveform(concat_mel.T)
            summary_manager.add_audio(f'speaker {test_speaker_id}/prediction', test_wavs[None, :, None],
                                      sr=summary_manager.config['sampling_rate'],
                                      step=summary_manager.global_step)
            
            ref_mel = ref_mel.numpy()[np.sum(tf.cast(ref_mel != 0, tf.uint8), axis=-1) != 0]
            test_reference_wav = summary_manager.audio.reconstruct_waveform(ref_mel.T)
            summary_manager.add_audio(f'speaker {test_speaker_id}/reference', test_reference_wav[None, :, None],
                                      sr=summary_manager.config['sampling_rate'],
                                      step=summary_manager.global_step)

print('Done.')
