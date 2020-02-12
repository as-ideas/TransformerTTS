import os

import tensorflow as tf

from utils.decorators import ignore_exception
from utils.display import plot_attention, display_mel


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

    @ignore_exception
    def write_images(self, mel, pred, step, id):
        for kind in self.mel_kinds:
            self._write_image(kind,
                              mel=mel,
                              pred=pred[kind],
                              step=step,
                              id=id)

    @ignore_exception
    def write_text(self, text, pred, step):
        for kind in self.text_kinds:
            pred_decoded = pred[kind]['output_decoded']
            self._write_text(kind, text, pred_decoded, step)

    @ignore_exception
    def write_loss(self, output, step, name='loss'):
        for kind in self.all_kinds:
            with self.summary_writers[kind].as_default():
                loss = output[kind]['loss']
                tf.summary.scalar(name, loss, step=step)
                if kind in self.mel_kinds and 'losses' in output[kind]:
                    for k in output[kind]['losses'].keys():
                        loss = output[kind]['losses'][k]
                        tf.summary.scalar(kind + '_' + k, loss, step=step)
                        
    # TODO: this is horrible, to individually double check lr, dropout, etc...
    @ignore_exception
    def write_meta_for_kind(self, name, value, step, kind):
        with self.summary_writers[kind].as_default():
            tf.summary.scalar(name, tf.Variable(value), step=step)
            
    @ignore_exception
    def write_meta(self, name, value, step):
        with self.summary_writers['meta'].as_default():
            tf.summary.scalar(name, tf.Variable(value), step=step)
    
    @ignore_exception
    def write_attention(self, output, step):
        for kind in self.all_kinds:
            with self.summary_writers[kind].as_default():
                plot_attention(outputs=output[kind],
                               step=step,
                               info_string='train attention ')
    
    def _write_image(self, kind, mel, pred, step, id):
        with self.summary_writers[kind].as_default():
            plot_attention(outputs=pred,
                           step=step,
                           info_string='test attention ')
            display_mel(mel=pred['mel'],
                        step=step,
                        info_string=f'test mel {id}')
            display_mel(mel=mel,
                        step=step,
                        info_string=f'target mel {id}')
    
    def _write_text(self, kind, text, pred_decoded, step):
        with self.summary_writers[kind].as_default():
            name = f'{kind} from validation'
            data_pred = f'(pred) {pred_decoded}'
            data_target = f'(target) {text}'
            tf.summary.text(name=name, data=f'{data_pred}\n{data_target}', step=step)
