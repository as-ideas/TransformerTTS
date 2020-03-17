import os

import tensorflow as tf

from utils.audio import invert_griffin_lim
from utils.decorators import ignore_exception
from utils.display import plot_attention, display_mel


class SummaryManager:
    
    def __init__(self,
                 log_dir):
        meta_path = os.path.join(log_dir, 'meta')
        self.summary_writers = {'model': tf.summary.create_file_writer(log_dir),
                                'meta': tf.summary.create_file_writer(meta_path)}

    @ignore_exception
    def write_images(self, mel, pred, step, id):
        with self.summary_writers['model'].as_default():
            plot_attention(outputs=pred,
                           step=step,
                           info_string='test attention ')
            display_mel(mel=pred['mel'],
                        step=step,
                        info_string=f'test mel {id}')
            display_mel(mel=mel,
                        step=step,
                        info_string=f'target mel {id}')

    @ignore_exception
    def write_text(self, text, pred, step):
        pred_decoded = pred['output_decoded']
        self._write_text(text, pred_decoded, step)

    @ignore_exception
    def write_loss(self, output, step, name='loss'):
        with self.summary_writers['model'].as_default():
            loss = output['loss']
            tf.summary.scalar(name, loss, step=step)
            for k in output['losses'].keys():
                loss = output['losses'][k]
                tf.summary.scalar('model_' + k, loss, step=step)
                
    @ignore_exception
    def write_audios(self, mel, pred, config, step, id):
        self._write_audio(
                          name=f'target {id}',
                          mel=mel,
                          config=config,
                          step=step)
        self._write_audio(
                          name=f'pred {id}',
                          mel=pred['mel'],
                          config=config,
                          step=step)

    @ignore_exception
    def write_meta_scalar(self, name, value, step):
        with self.summary_writers['meta'].as_default():
            tf.summary.scalar(name, tf.Variable(value), step=step)

    @ignore_exception
    def write_attention(self, output, step):
        with self.summary_writers['model'].as_default():
            plot_attention(outputs=output,
                           step=step,
                           info_string='train attention ')
    
    def _write_image(self, mel, pred, step, id):
        with self.summary_writers['model'].as_default():
            plot_attention(outputs=pred,
                           step=step,
                           info_string='test attention ')
            display_mel(mel=pred['mel'],
                        step=step,
                        info_string=f'test mel {id}')
            display_mel(mel=mel,
                        step=step,
                        info_string=f'target mel {id}')
    
    def _write_text(self, text, pred_decoded, step):
        with self.summary_writers['model'].as_default():
            name = f'TTS from validation'
            data_pred = f'(pred) {pred_decoded}'
            data_target = f'(target) {text}'
            tf.summary.text(name=name, data=f'{data_pred}\n{data_target}', step=step)

    def _write_audio(self, name, mel, config, step):
        wav = invert_griffin_lim(mel, config)
        wav = tf.expand_dims(wav, 0)
        wav = tf.expand_dims(wav, -1)
        with self.summary_writers['model'].as_default():
            tf.summary.audio(name,
                             wav,
                             sample_rate=config['sampling_rate'],
                             step=step)