import argparse
import pickle

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from p_tqdm import p_umap

from utils.config_manager import Config
from utils.logging_utils import SummaryManager
from data.datasets import AlignerPreprocessor
from utils.alignments import get_durations_from_alignment
from utils.scripts_utils import dynamic_memory_allocation
from data.datasets import AlignerDataset
from data.datasets import DataReader

np.random.seed(42)
tf.random.set_seed(42)
dynamic_memory_allocation()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', type=str)
    parser.add_argument('--best', dest='best', action='store_true',
                        help='Use best head instead of weighted average of heads.')
    parser.add_argument('--autoregressive_weights', type=str, default=None,
                        help='Explicit path to autoregressive model weights.')
    parser.add_argument('--skip_char_pitch', dest='skip_char_pitch', action='store_true')
    parser.add_argument('--skip_durations', dest='skip_durations', action='store_true')
    args = parser.parse_args()
    weighted = not args.best
    tag_description = ''.join([
        f'{"_weighted" * weighted}{"_best" * (not weighted)}',
    ])
    writer_tag = f'DurationExtraction{tag_description}'
    print(writer_tag)
    config_manager = Config(config_path=args.config, model_kind='aligner')
    config = config_manager.config
    config_manager.print_config()
    
    if not args.skip_durations:
        model = config_manager.load_model(args.autoregressive_weights)
        if model.r != 1:
            print(f"ERROR: model's reduction factor is greater than 1, check config. (r={model.r}")
        
        data_prep = AlignerPreprocessor.from_config(config=config_manager,
                                                    tokenizer=model.text_pipeline.tokenizer)
        data_handler = AlignerDataset.from_config(config_manager,
                                                  preprocessor=data_prep,
                                                  kind='phonemized')
        target_dir = config_manager.duration_dir
        target_dir.mkdir(exist_ok=True)
        config_manager.dump_config()
        dataset = data_handler.get_dataset(bucket_batch_sizes=config['bucket_batch_sizes'],
                                           bucket_boundaries=config['bucket_boundaries'],
                                           shuffle=False,
                                           drop_remainder=False)
        
        last_layer_key = 'Decoder_LastBlock_CrossAttention'
        print(f'Extracting attention from layer {last_layer_key}')
        
        summary_manager = SummaryManager(model=model, log_dir=config_manager.log_dir / 'Duration Extraction',
                                         config=config,
                                         default_writer='Duration Extraction')
        all_durations = np.array([])
        new_alignments = []
        iterator = tqdm(enumerate(dataset.all_batches()))
        step = 0
        for c, (mel_batch, text_batch, stop_batch, file_name_batch, tar_mel) in iterator:
            iterator.set_description(f'Processing dataset')
            outputs = model.val_step(inp=text_batch,
                                     tar=mel_batch,
                                     stop_prob=stop_batch,
                                     tar_mel=tar_mel)
            attention_values = outputs['decoder_attention'][last_layer_key].numpy()
            text = text_batch.numpy()
            
            mel = mel_batch.numpy()
            
            durations, final_align, jumpiness, peakiness, diag_measure = get_durations_from_alignment(
                batch_alignments=attention_values,
                mels=mel,
                phonemes=text,
                weighted=weighted)
            batch_avg_jumpiness = tf.reduce_mean(jumpiness, axis=0)
            batch_avg_peakiness = tf.reduce_mean(peakiness, axis=0)
            batch_avg_diag_measure = tf.reduce_mean(diag_measure, axis=0)
            for i in range(tf.shape(jumpiness)[1]):
                summary_manager.display_scalar(tag=f'DurationAttentionJumpiness/head{i}',
                                               scalar_value=tf.reduce_mean(batch_avg_jumpiness[i]), step=c)
                summary_manager.display_scalar(tag=f'DurationAttentionPeakiness/head{i}',
                                               scalar_value=tf.reduce_mean(batch_avg_peakiness[i]), step=c)
                summary_manager.display_scalar(tag=f'DurationAttentionDiagonality/head{i}',
                                               scalar_value=tf.reduce_mean(batch_avg_diag_measure[i]), step=c)
            
            for i, name in enumerate(file_name_batch):
                all_durations = np.append(all_durations, durations[i])  # for plotting only
                summary_manager.add_image(tag='ExtractedAlignments',
                                          image=tf.expand_dims(tf.expand_dims(final_align[i], 0), -1),
                                          step=step)
                
                step += 1
                np.save(str(target_dir / f"{name.numpy().decode('utf-8')}.npy"), durations[i])
        
        all_durations[all_durations >= 20] = 20  # for plotting only
        buckets = len(set(all_durations))  # for plotting only
        summary_manager.add_histogram(values=all_durations, tag='ExtractedDurations', buckets=buckets)
    
    if not args.skip_char_pitch:
        def _pitch_per_char(pitch, durations, mel_len):
            durs_cum = np.cumsum(np.pad(durations, (1, 0)))
            pitch_char = np.zeros((durations.shape[0],), dtype=np.float)
            for idx, a, b in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
                values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
                values = values[np.where((values * pitch_stats['pitch_std'] + pitch_stats['pitch_mean']) < 400)[0]]
                pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0
            return pitch_char
        
        
        def process_per_char_pitch(sample_name: str):
            pitch = np.load((config_manager.pitch_dir / sample_name).with_suffix('.npy').as_posix())
            durations = np.load((config_manager.duration_dir / sample_name).with_suffix('.npy').as_posix())
            mel = np.load((config_manager.mel_dir / sample_name).with_suffix('.npy').as_posix())
            # text = metadatareader.text_dict[sample_name]
            char_wise_pitch = _pitch_per_char(pitch, durations, mel.shape[0])
            # len_text = len(text)
            # if config_manager.config['initial_breathing']:
            #     len_text += 1 # TODO: improve this, BREATHING TOKEN needs +1
            # assert char_wise_pitch.shape[0] == len_text, \
            #     f'{sample_name}: dshape {char_wise_pitch.shape} == tshape {len_text}'
            np.save((config_manager.pitch_per_char / sample_name).with_suffix('.npy').as_posix(), char_wise_pitch)
        
        
        metadatareader = DataReader.from_config(config_manager, kind='phonemized', scan_wavs=False)
        pitch_stats = pickle.load(open(config_manager.data_dir / 'pitch_stats.pkl', 'rb'))
        print(f'\nComputing phoneme-wise pitch')
        print(f'{len(metadatareader.filenames)} items found in {metadatareader.metadata_path}.')
        wav_iter = p_umap(process_per_char_pitch, metadatareader.filenames)
    print('Done.')
