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

np.random.seed(42)
tf.random.set_seed(42)

dynamic_memory_allocation()
parser = basic_train_parser()
args = parser.parse_args()


@ignore_exception
@time_it
def validate(model,
             val_dataset,
             summary_manager):
    val_loss = {'loss': 0.}
    norm = 0.
    for val_mel, val_text, val_stop, fname in val_dataset.all_batches():
        model_out = model.val_step(inp=val_text,
                                   tar=val_mel,
                                   stop_prob=val_stop)
        norm += 1
        val_loss['loss'] += model_out['loss']
    val_loss['loss'] /= norm
    summary_manager.display_loss(model_out, tag='Validation', plot_all=True)
    summary_manager.display_attention_heads(model_out, tag='ValidationAttentionHeads')
    # summary_manager.display_mel(mel=model_out['mel_linear'][0], tag=f'Validation/linear_mel_out')
    summary_manager.display_mel(mel=model_out['final_output'][0],
                                tag=f'Validation/predicted_mel_{fname[0].numpy().decode("utf-8")}')
    # residual = abs(model_out['mel_linear'] - model_out['final_output'])
    # summary_manager.display_mel(mel=residual[0], tag=f'Validation/conv-linear_residual')
    summary_manager.display_mel(mel=val_mel[0], tag=f'Validation/target_mel_{fname[0].numpy().decode("utf-8")}')
    return val_loss['loss']


config_manager = Config(config_path=args.config, model_kind='autoregressive')
config = config_manager.config
config_manager.create_remove_dirs(clear_dir=args.clear_dir,
                                  clear_logs=args.clear_logs,
                                  clear_weights=args.clear_weights)
config_manager.dump_config()
config_manager.print_config()
#


# get model, prepare data for model, create datasets
model = config_manager.get_model()
config_manager.compile_model(model)
data_prep = AutoregressivePreprocessor.from_config(config_manager,
                                                   tokenizer=model.text_pipeline.tokenizer)
train_data_handler = TextMelDataset.from_config(config_manager,
                                                preprocessor=data_prep,
                                                kind='train')
valid_data_handler = TextMelDataset.from_config(config_manager,
                                                preprocessor=data_prep,
                                                kind='valid')

train_dataset = train_data_handler.get_dataset(bucket_batch_sizes=config['bucket_batch_sizes'],
                                               bucket_boundaries=config['bucket_boundaries'],
                                               shuffle=True)
valid_dataset = valid_data_handler.get_dataset(bucket_batch_sizes=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1],
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
test_mel, test_phonemes, test_stop, test_fname = valid_dataset.next_batch()
_ = train_dataset.next_batch()
t = trange(model.step, config['max_steps'], leave=True)
for _ in t:
    t.set_description(f'step {model.step}')
    mel, phonemes, stop, sample_name = train_dataset.next_batch()
    decoder_prenet_dropout = piecewise_linear_schedule(model.step, config['decoder_prenet_dropout_schedule'])
    learning_rate = piecewise_linear_schedule(model.step, config['learning_rate_schedule'])
    reduction_factor = reduction_schedule(model.step, config['reduction_factor_schedule'])
    drop_n_heads = tf.cast(reduction_schedule(model.step, config['head_drop_schedule']), tf.int32)
    t.display(f'reduction factor {reduction_factor}', pos=10)
    model.set_constants(decoder_prenet_dropout=decoder_prenet_dropout,
                        learning_rate=learning_rate,
                        reduction_factor=reduction_factor,
                        drop_n_heads=drop_n_heads)
    output = model.train_step(inp=phonemes,
                              tar=mel,
                              stop_prob=stop)
    losses.append(float(output['loss']))
    
    t.display(f'step loss: {losses[-1]}', pos=1)
    for pos, n_steps in enumerate(config['n_steps_avg_losses']):
        if len(losses) > n_steps:
            t.display(f'{n_steps}-steps average loss: {sum(losses[-n_steps:]) / n_steps}', pos=pos + 2)
    
    summary_manager.display_loss(output, tag='Train')
    summary_manager.display_scalar(tag='Meta/decoder_prenet_dropout', scalar_value=model.decoder_prenet.rate)
    summary_manager.display_scalar(tag='Meta/learning_rate', scalar_value=model.optimizer.lr)
    summary_manager.display_scalar(tag='Meta/reduction_factor', scalar_value=model.r)
    summary_manager.display_scalar(tag='Meta/drop_n_heads', scalar_value=model.drop_n_heads)
    if model.step % config['train_images_plotting_frequency'] == 0:
        summary_manager.display_attention_heads(output, tag='TrainAttentionHeads')
        summary_manager.display_mel(mel=output['mel_linear'][0], tag=f'Train/linear_mel_out')
        summary_manager.display_mel(mel=output['final_output'][0], tag=f'Train/predicted_mel')
        residual = abs(output['mel_linear'] - output['final_output'])
        summary_manager.display_mel(mel=residual[0], tag=f'Train/conv-linear_residual')
        summary_manager.display_mel(mel=mel[0], tag=f'Train/target_mel')
        summary_manager.display_audio(tag=f'Train/prediction', mel=output['final_output'][0])
        summary_manager.display_audio(tag=f'Train/target', mel=mel[0])
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
    
    if model.step % config['validation_frequency'] == 0:
        val_loss, time_taken = validate(model=model,
                                        val_dataset=valid_dataset,
                                        summary_manager=summary_manager)
        t.display(f'validation loss at step {model.step}: {val_loss} (took {time_taken}s)',
                  pos=len(config['n_steps_avg_losses']) + 3)
    
    if model.step % config['prediction_frequency'] == 0 and (model.step >= config['prediction_start_step']):
        for j in range(len(test_mel)):
            if j < config['n_predictions']:
                mel, phonemes, stop, fname = test_mel[j], test_phonemes[j], test_stop[j], test_fname[j]
                mel = mel[tf.reduce_sum(tf.cast(mel != 0, tf.int32), axis=1) > 0]
                t.display(f'Predicting {j}', pos=len(config['n_steps_avg_losses']) + 4)
                pred = model.predict(phonemes,
                                     max_length=mel.shape[0] + 50,
                                     encode=False,
                                     verbose=False)
                pred_mel = pred['mel']
                mel = mel[1:-1]
                target_mel = mel
                summary_manager.display_attention_heads(outputs=pred,
                                                        tag=f'TestAttentionHeads/{fname.numpy().decode("utf-8")}')
                summary_manager.display_mel(mel=pred_mel, tag=f'Test/{fname.numpy().decode("utf-8")}/predicted')
                summary_manager.display_mel(mel=target_mel, tag=f'Test/{fname.numpy().decode("utf-8")}/target')
                if model.step >= config['audio_start_step']:
                    summary_manager.display_audio(tag=f'{fname.numpy().decode("utf-8")}/target', mel=target_mel)
                    summary_manager.display_audio(tag=f'{fname.numpy().decode("utf-8")}/prediction', mel=pred_mel)

print('Done.')
