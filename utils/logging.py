from pathlib import Path

import tensorflow as tf

from utils.audio import reconstruct_waveform, denormalize
from utils.display import buffer_mel, tight_grid
from utils.vec_ops import norm_tensor
from utils.decorators import ignore_exception


def control_frequency(f):
    def apply_func(*args, **kwargs):
        # args[0] is self
        plot_all = ('plot_all' in kwargs) and kwargs['plot_all']
        if (args[0].global_step % args[0].plot_frequency == 0) or plot_all:
            result = f(*args, **kwargs)
            return result
        else:
            return None
    
    return apply_func


class SummaryManager:
    """ Writes tensorboard logs during training.
    
        :arg model: model object that is trained
        :arg log_dir: base directory where logs of a config are created
        :arg config: configuration dictionary
        :arg max_plot_frequency: every how many steps to plot
    """
    
    def __init__(self,
                 model,
                 log_dir,
                 config,
                 max_plot_frequency=10):
        self.model = model
        self.log_dir = Path(log_dir)
        self.config = config
        self.plot_frequency = max_plot_frequency
        self.writers = {'log_dir': tf.summary.create_file_writer(str(self.log_dir))}
    
    def add_writer(self, path):
        """ Adds a writer to self.writers if the writer does not exist already.
            To avoid spamming writers on disk.
            
            :returns the writer with path as tag
        """
        if path not in self.writers.keys():
            self.writers[path] = tf.summary.create_file_writer(str(path))
        return self.writers[path]
    
    @property
    def global_step(self):
        return self.model.step
    
    def add_scalars(self, tag, dictionary):
        for k in dictionary.keys():
            with self.add_writer(str(self.log_dir / k)).as_default():
                tf.summary.scalar(name=tag, data=dictionary[k], step=self.global_step)
    
    def add_scalar(self, tag, scalar_value):
        with self.writers['log_dir'].as_default():
            tf.summary.scalar(name=tag, data=scalar_value, step=self.global_step)
    
    def add_image(self, tag, image):
        with self.writers['log_dir'].as_default():
            tf.summary.image(name=tag, data=image, step=self.global_step, max_outputs=4)
    
    def add_histogram(self, tag, values):
        with self.writers['log_dir'].as_default():
            tf.summary.histogram(name=tag, data=values, step=self.global_step)
    
    def add_audio(self, tag, wav, sr):
        with self.writers['log_dir'].as_default():
            tf.summary.audio(name=tag,
                             data=wav,
                             sample_rate=sr,
                             step=self.global_step)
    
    @ignore_exception
    def display_attention_heads(self, outputs, tag=''):
        for layer in ['encoder_attention', 'decoder_attention']:
            for k in outputs[layer].keys():
                image = tight_grid(norm_tensor(outputs[layer][k][0]))
                # dim 0 of image_batch is now number of heads
                batch_plot_path = f'{tag}/{layer}/{k}'
                self.add_image(str(batch_plot_path), tf.expand_dims(tf.expand_dims(image, 0), -1))

    @ignore_exception
    def display_mel(self, mel, tag='', sr=22050):
        amp_mel = denormalize(mel, self.config)
        img = tf.transpose(amp_mel)
        buf = buffer_mel(img, sr=sr)
        img_tf = tf.image.decode_png(buf.getvalue(), channels=3)
        self.add_image(tag, tf.expand_dims(img_tf, 0))
    
    @control_frequency
    @ignore_exception
    def display_loss(self, output, tag='', plot_all=False):
        self.add_scalars(tag=f'{tag}/losses', dictionary=output['losses'])
        self.add_scalar(tag=f'{tag}/loss', scalar_value=output['loss'])
    
    @control_frequency
    @ignore_exception
    def display_scalar(self, tag, scalar_value, plot_all=False):
        self.add_scalar(tag=tag, scalar_value=scalar_value)
    
    @ignore_exception
    def display_audio(self, tag, mel):
        wav = reconstruct_waveform(tf.transpose(mel), self.config)
        wav = tf.expand_dims(wav, 0)
        wav = tf.expand_dims(wav, -1)
        self.add_audio(tag, wav.numpy(), sr=self.config['sampling_rate'])
