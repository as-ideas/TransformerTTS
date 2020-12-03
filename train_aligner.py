import tensorflow as tf
import numpy as np
from tqdm import trange

from utils.config_manager import Config
from preprocessing.datasets import TextMelDataset, AutoregressivePreprocessor
from utils.decorators import ignore_exception, time_it
from utils.scheduling import piecewise_linear_schedule, reduction_schedule
from utils.logging_utils import SummaryManager
from utils.scripts_utils import dynamic_memory_allocation, basic_train_parser
from utils.metrics import attention_score
from utils.spectrogram_ops import mel_lengths, phoneme_lengths
from utils.alignments import get_durations_from_alignment

np.random.seed(42)
tf.random.set_seed(42)

dynamic_memory_allocation()
parser = basic_train_parser()
args = parser.parse_args()


def cut_with_durations(durations, mel, phonemes, snippet_len=10):
    phon_dur = np.pad(durations, (1, 0))
    starts = np.cumsum(phon_dur)[:-1]
    ends = np.cumsum(phon_dur)[1:]
    cut_mels = []
    cut_texts = []
    for end_idx in range(snippet_len, len(phon_dur), snippet_len):
        start_idx = end_idx - snippet_len
        cut_mels.append(mel[starts[start_idx]: ends[end_idx - 1], :])
        cut_texts.append(phonemes[start_idx: end_idx])
    return cut_mels, cut_texts


@ignore_exception
@time_it
def validate(model,
             val_dataset,
             summary_manager,
             weighted_durations):
    val_loss = {'loss': 0.}
    norm = 0.
    current_r = model.r
    model.set_constants(reduction_factor=1)
    for val_mel, val_text, val_stop, fname, tar_mel in val_dataset.all_batches():
        model_out = model.val_step(inp=val_text,
                                   tar=val_mel,
                                   stop_prob=val_stop,
                                   tar_mel=tar_mel)
        norm += 1
        val_loss['loss'] += model_out['loss']
    val_loss['loss'] /= norm
    summary_manager.display_loss(model_out, tag='Validation', plot_all=True)
    summary_manager.display_last_attention(model_out, tag='ValidationAttentionHeads', fname=fname)
    attention_values = model_out['decoder_attention']['Decoder_LastBlock_CrossAttention'].numpy()
    text = val_text.numpy()
    mel = val_mel.numpy()
    model.set_constants(reduction_factor=current_r)
    modes = list({False, weighted_durations})
    for mode in modes:
        durations, final_align, jumpiness, peakiness, diag_measure = get_durations_from_alignment(
            batch_alignments=attention_values,
            mels=mel,
            phonemes=text,
            weighted=mode)
        for k in range(len(durations)):
            phon_dur = durations[k]
            imel = mel[k][1:]  # remove start token (is padded so end token can't be removed/not an issue)
            itext = text[k][1:]  # remove start token (is padded so end token can't be removed/not an issue)
            iphon = model.text_pipeline.tokenizer.decode(itext)
            cut_mels, cut_texts = cut_with_durations(durations=phon_dur, mel=imel, phonemes=iphon)
            for cut_idx, cut_text in enumerate(cut_texts):
                weighted_label = 'weighted_' * mode
                summary_manager.display_audio(
                    tag=f'CutAudio_{weighted_label}{fname[k].numpy().decode("utf-8")}_{iphon}/{cut_idx}/{cut_text}',
                    mel=cut_mels[cut_idx], description=iphon)
    return val_loss['loss']


config_manager = Config(config_path=args.config, model_kind='aligner')
config = config_manager.config
config_manager.create_remove_dirs(clear_dir=args.clear_dir,
                                  clear_logs=args.clear_logs,
                                  clear_weights=args.clear_weights)
config_manager.dump_config()
config_manager.print_config()

# get model, prepare data for model, create datasets
model = config_manager.get_model()
config_manager.compile_model(model)
data_prep = AutoregressivePreprocessor.from_config(config_manager,
                                                   tokenizer=model.text_pipeline.tokenizer)  # TODO: tokenizer is now static
train_data_handler = TextMelDataset.from_config(config_manager,
                                                preprocessor=data_prep,
                                                kind='train')
valid_data_handler = TextMelDataset.from_config(config_manager,
                                                preprocessor=data_prep,
                                                kind='valid')

train_dataset = train_data_handler.get_dataset(bucket_batch_sizes=config['bucket_batch_sizes'],
                                               bucket_boundaries=config['bucket_boundaries'],
                                               shuffle=True)
valid_dataset = valid_data_handler.get_dataset(bucket_batch_sizes=config['val_bucket_batch_size'],
                                               bucket_boundaries=config['bucket_boundaries'],
                                               shuffle=False, drop_remainder=True)

# create logger and checkpointer and restore latest model

summary_manager = SummaryManager(model=model, log_dir=config_manager.log_dir, config=config)
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 optimizer=model.optimizer,
                                 net=model)
manager = tf.train.CheckpointManager(checkpoint, str(config_manager.weights_dir),
                                     max_to_keep=config['keep_n_weights'],
                                     keep_checkpoint_every_n_hours=config['keep_checkpoint_every_n_hours'])
manager_training = tf.train.CheckpointManager(checkpoint, str(config_manager.weights_dir / 'latest'),
                                              max_to_keep=1, checkpoint_name='latest')

checkpoint.restore(manager_training.latest_checkpoint)
if manager_training.latest_checkpoint:
    print(f'\nresuming training from step {model.step} ({manager_training.latest_checkpoint})')
else:
    print(f'\nstarting training from scratch')

if config['debug'] is True:
    print('\nWARNING: DEBUG is set to True. Training in eager mode.')
# main event
print('\nTRAINING')
losses = []
test_mel, test_phonemes, test_stop, test_fname, test_tar_mel = valid_dataset.next_batch()
_ = train_dataset.next_batch()
t = trange(model.step, config['max_steps'], leave=True)
for _ in t:
    t.set_description(f'step {model.step}')
    mel, phonemes, stop, sample_name, tar_mel = train_dataset.next_batch()
    learning_rate = piecewise_linear_schedule(model.step, config['learning_rate_schedule'])
    reduction_factor = reduction_schedule(model.step, config['reduction_factor_schedule'])
    drop_n_heads = tf.cast(reduction_schedule(model.step, config['head_drop_schedule']), tf.int32)
    t.display(f'reduction factor {reduction_factor}', pos=10)
    if model.step < config['force_encoder_diagonal_steps']:
        force_encoder_diagonal = True
    else:
        force_encoder_diagonal = False
    if model.step < config['force_decoder_diagonal_steps']:
        force_decoder_diagonal = True
    else:
        force_decoder_diagonal = False
    model.set_constants(learning_rate=learning_rate,
                        reduction_factor=reduction_factor,
                        drop_n_heads=drop_n_heads,
                        force_encoder_diagonal=force_encoder_diagonal,
                        force_decoder_diagonal=force_decoder_diagonal)
    
    output = model.train_step(inp=phonemes,
                              tar=mel,
                              stop_prob=stop,
                              tar_mel=tar_mel)
    losses.append(float(output['loss']))
    
    t.display(f'step loss: {losses[-1]}', pos=1)
    for pos, n_steps in enumerate(config['n_steps_avg_losses']):
        if len(losses) > n_steps:
            t.display(f'{n_steps}-steps average loss: {sum(losses[-n_steps:]) / n_steps}', pos=pos + 2)
    
    summary_manager.display_loss(output, tag='Train')
    summary_manager.display_scalar(tag='Meta/learning_rate', scalar_value=model.optimizer.lr)
    summary_manager.display_scalar(tag='Meta/reduction_factor', scalar_value=model.r)
    summary_manager.display_scalar(tag='Meta/drop_n_heads', scalar_value=model.drop_n_heads)
    summary_manager.display_scalar(scalar_value=t.avg_time, tag='Meta/iter_time')
    summary_manager.display_scalar(scalar_value=tf.shape(sample_name)[0], tag='Meta/batch_size')
    if model.step % config['train_images_plotting_frequency'] == 0:
        summary_manager.display_attention_heads(output, tag='TrainAttentionHeads')
        summary_manager.display_mel(mel=output['mel'][0], tag=f'Train/predicted_mel')
        summary_manager.display_mel(mel=tar_mel[0], tag=f'Train/target_mel')
        for layer, k in enumerate(output['decoder_attention'].keys()):
            mel_lens = mel_lengths(mel_batch=mel, padding_value=0) // model.r  # [N]
            phon_len = phoneme_lengths(phonemes)
            loc_score, peak_score, diag_measure = attention_score(att=output['decoder_attention'][k],
                                                                  mel_len=mel_lens,
                                                                  phon_len=phon_len,
                                                                  r=model.r)
            loc_score = tf.reduce_mean(loc_score, axis=0)
            peak_score = tf.reduce_mean(peak_score, axis=0)
            diag_measure = tf.reduce_mean(diag_measure, axis=0)
            for i in range(tf.shape(loc_score)[0]):
                summary_manager.display_scalar(tag=f'TrainDecoderAttentionJumpiness/layer{layer}_head{i}',
                                               scalar_value=tf.reduce_mean(loc_score[i]))
                summary_manager.display_scalar(tag=f'TrainDecoderAttentionPeakiness/layer{layer}_head{i}',
                                               scalar_value=tf.reduce_mean(peak_score[i]))
                summary_manager.display_scalar(tag=f'TrainDecoderAttentionDiagonality/layer{layer}_head{i}',
                                               scalar_value=tf.reduce_mean(diag_measure[i]))
    
    if model.step % 1000 == 0:
        save_path = manager_training.save()
    if model.step % config['weights_save_frequency'] == 0:
        save_path = manager.save()
        t.display(f'checkpoint at step {model.step}: {save_path}', pos=len(config['n_steps_avg_losses']) + 2)
    
    if model.step % config['validation_frequency'] == 0 and (model.step >= config['prediction_start_step']):
        val_loss, time_taken = validate(model=model,
                                        val_dataset=valid_dataset,
                                        summary_manager=summary_manager,
                                        weighted_durations=config['extract_attention_weighted'])
        t.display(f'validation loss at step {model.step}: {val_loss} (took {time_taken}s)',
                  pos=len(config['n_steps_avg_losses']) + 3)
print('Done.')
