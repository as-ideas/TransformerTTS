import os
import datetime
import argparse
from pathlib import Path

import tensorflow as tf
import numpy as np

from losses import masked_crossentropy, masked_mean_squared_error
from model.transformer_factory import new_everything
from utils import plot_mel, buffer_mel

np.random.seed(42)
tf.random.set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--mel_dir', dest='MEL_DIR', type=str)
parser.add_argument('--n_samples', dest='MAX_SAMPLES', default=300, type=int)
parser.add_argument('--dropout', dest='DROPOUT', default=0.1, type=float)
parser.add_argument('--noisestd', dest='NOISE_STD', default=0.2, type=float)
parser.add_argument('--mel', dest='MEL_CHANNELS', default=80, type=int)
parser.add_argument('--epochs', dest='EPOCHS', default=10000, type=int)
parser.add_argument('--batch_size', dest='BATCH_SIZE', default=16, type=int)
parser.add_argument('--text_freq', dest='TEXT_FREQ', default=1000, type=int)
parser.add_argument('--image_freq', dest='IMAGE_FREQ', default=10, type=int)
parser.add_argument('--learning_rate', dest='LEARNING_RATE', default=1e-4, type=float)
args = parser.parse_args()

mel_path = Path(args.MEL_DIR) / 'mels'
metafile = Path(args.MEL_DIR) / 'train_metafile.txt'

sr = 22050
N_EPOCHS = args.EPOCHS
N_SAMPLES = args.MAX_SAMPLES
image_freq = args.IMAGE_FREQ
text_freq = args.TEXT_FREQ
num_layers = 4
d_model = 256
num_heads = 8
dff = 512
# TODO: differentiate
# DFF_TEXT = 256
dff_prenet = 256
max_position_encoding = 1000
dropout_rate = args.DROPOUT
stop_prob_index = 2
postnet_conv_filters = 256
postnet_conv_layers = 5
postnet_kernel_size = 5

mel_channels = args.MEL_CHANNELS


class TestTokenizer:
    
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet, start=1)}
        self.idx_to_token[0] = '/'
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.start_token_index = len(self.alphabet) + 1
        self.end_token_index = len(self.alphabet) + 2
        self.vocab_size = len(self.alphabet) + 3
        self.idx_to_token[self.start_token_index] = '<'
        self.idx_to_token[self.end_token_index] = '>'
    
    def encode(self, sentence):
        return [self.token_to_idx[c] for c in sentence]
    
    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence])


def norm_tensor(tensor):
    return tf.math.divide(
        tf.math.subtract(
            tensor,
            tf.math.reduce_min(tensor)
        ),
        tf.math.subtract(
            tf.math.reduce_max(tensor),
            tf.math.reduce_min(tensor)
        )
    )


def plot_attention(outputs, step, info_string=''):
    for k in outputs['attention_weights'].keys():
        for i in range(len(outputs['attention_weights'][k][0])):
            image_batch = norm_tensor(tf.expand_dims(outputs['attention_weights'][k][:, i, :, :], -1))
            tf.summary.image(info_string + k + f' head{i}', image_batch,
                             step=step)


def display_mel(pred, step, info_string='', sr=22050):
    img = tf.transpose(tf.exp(pred))
    buf = buffer_mel(img, sr=sr)
    img_tf = tf.image.decode_png(buf.getvalue(), channels=3)
    img_tf = tf.expand_dims(img_tf, 0)
    tf.summary.image(info_string, img_tf, step=step)


def get_norm_mel(mel_path, start_vec, end_vec):
    mel = np.load(mel_path)
    norm_mel = np.log(mel.clip(1e-5))
    norm_mel = np.concatenate([start_vec, norm_mel, end_vec])
    return norm_mel


start_vec = np.ones((1, mel_channels)) * -3
end_vec = np.ones((1, mel_channels))

mel_text_stop_samples = []
count = 0
alphabet = set()
with open(str(metafile), 'r', encoding='utf-8') as f:
    for l in f.readlines():
        l_split = l.split('|')
        text = l_split[-1].strip().lower()
        mel_file = os.path.join(str(mel_path), l_split[0] + '.npy')
        norm_mel = get_norm_mel(mel_file, start_vec, end_vec)
        stop_probs = np.ones(norm_mel.shape[0], dtype=np.int64)
        stop_probs[-1] = 2
        mel_text_stop_samples.append((norm_mel, text, stop_probs))
        alphabet.update(list(text))
        count += 1
        if count > N_SAMPLES:
            break

tokenizer = TestTokenizer(alphabet=sorted(list(alphabet)))
start_tok, end_tok = tokenizer.start_token_index, tokenizer.end_token_index
tokenized_mel_text_train_samples = [(mel, [start_tok] + tokenizer.encode(text) + [end_tok], stop_prob)
                                    for mel, text, stop_prob in mel_text_stop_samples]
mel_text_stop_gen = lambda: (pair for pair in tokenized_mel_text_train_samples[10:])
mel_text_stop_dataset = tf.data.Dataset.from_generator(mel_text_stop_gen,
                                                       output_types=(tf.float64, tf.int64, tf.int64))
mel_text_stop_dataset = mel_text_stop_dataset.shuffle(
    len(tokenized_mel_text_train_samples) // args.BATCH_SIze).padded_batch(args.BATCH_SIZE,
                                                                           padded_shapes=([-1, 80], [-1], [-1]),
                                                                           drop_remainder=True)
#mel_text_stop_dataset = mel_text_stop_dataset.prefetch(tf.data.experimental.AUTOTUNE)

input_vocab_size = tokenizer.vocab_size
target_vocab_size = tokenizer.vocab_size
start_token_index = tokenizer.start_token_index
end_token_index = tokenizer.end_token_index

transformers = new_everything(start_vec=start_vec,
                              stop_prob_index=stop_prob_index,
                              input_vocab_size=input_vocab_size,
                              start_token_index=start_token_index,
                              end_token_index=end_token_index,
                              target_vocab_size=target_vocab_size,
                              mel_channels=mel_channels,
                              num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              dff_prenet=dff_prenet,
                              max_position_encoding=max_position_encoding,
                              postnet_conv_filters=postnet_conv_filters,
                              postnet_conv_layers=postnet_conv_layers,
                              postnet_kernel_size=postnet_kernel_size,
                              dropout_rate=dropout_rate)
loss_coeffs = [1.0, 1.0, 1.0]
transformers['mel_to_text'].compile(loss=masked_crossentropy,
                                    optimizer=tf.keras.optimizers.Adam(args.LEARNING_RATE, beta_1=0.9, beta_2=0.98,
                                                                       epsilon=1e-9))
transformers['text_to_text'].compile(loss=masked_crossentropy,
                                     optimizer=tf.keras.optimizers.Adam(args.LEARNING_RATE, beta_1=0.9, beta_2=0.98,
                                                                        epsilon=1e-9))
transformers['mel_to_mel'].compile(loss=[masked_mean_squared_error,
                                         masked_crossentropy,
                                         masked_mean_squared_error],
                                   loss_weights=loss_coeffs,
                                   optimizer=tf.keras.optimizers.Adam(args.LEARNING_RATE, beta_1=0.9, beta_2=0.98,
                                                                      epsilon=1e-9))
transformers['text_to_mel'].compile(loss=[masked_mean_squared_error,
                                          masked_crossentropy,
                                          masked_mean_squared_error],
                                    loss_weights=loss_coeffs,
                                    optimizer=tf.keras.optimizers.Adam(args.LEARNING_RATE, beta_1=0.9, beta_2=0.98,
                                                                       epsilon=1e-9))
batch_count = 0
losses = {}
summary_writers = {}

weights_paths = {}
kinds = ['text_to_text', 'mel_to_mel', 'text_to_mel', 'mel_to_text']
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(f'/tmp/weights/train/{current_time}/', exist_ok=False)
for kind in kinds:
    summary_writers[kind] = tf.summary.create_file_writer(f'/tmp/summaries/train/{current_time}/{kind}')
    weights_paths[kind] = f'/tmp/weights/train/{current_time}/{kind}_weights.hdf5'
    losses[kind] = []


def linear_dropout_schedule(step):
    dout = max(((-0.9 + 0.5) / 20000.) * step + 0.9, .5)
    return tf.cast(dout, tf.float32)


for epoch in range(N_EPOCHS):
    for (batch, (mel, text, stop)) in enumerate(mel_text_stop_dataset):
        output = {}
        decoder_prenet_dropout = linear_dropout_schedule(batch_count)
        output['text_to_text'] = transformers['text_to_text'].train_step(text, text)
        output['mel_to_mel'] = transformers['mel_to_mel'].train_step(mel, mel, stop,
                                                                     decoder_prenet_dropout=decoder_prenet_dropout)
        output['text_to_mel'] = transformers['text_to_mel'].train_step(text, mel, stop,
                                                                       decoder_prenet_dropout=decoder_prenet_dropout)
        output['mel_to_text'] = transformers['mel_to_text'].train_step(mel, text)
        with summary_writers['text_to_text'].as_default():
            tf.summary.scalar('dropout', decoder_prenet_dropout, step=transformers['text_to_text'].optimizer.iterations)
        for kind in kinds:
            losses[kind].append(float(output[kind]['loss']))
        if batch_count % image_freq == 0:
            for kind in kinds:
                with summary_writers[kind].as_default():
                    plot_attention(output[kind], step=transformers[kind].optimizer.iterations,
                                   info_string='train attention ')
                transformers[kind].save_weights(weights_paths[kind])
            pred = {}
            test_val = {}
            for i in range(0, 3):
                mel_target = mel_text_stop_samples[i][0]
                max_pred_len = mel_text_stop_samples[i][0].shape[0] + 50
                test_val['text_to_mel'] = tokenizer.encode(mel_text_stop_samples[i][1])
                test_val['mel_to_mel'] = mel_target
                for kind in ['text_to_mel', 'mel_to_mel']:
                    pred[kind] = transformers[kind].predict(test_val[kind],
                                                            max_length=max_pred_len,
                                                            decoder_prenet_dropout=0.5)
                    with summary_writers[kind].as_default():
                        plot_attention(pred[kind], step=transformers[kind].optimizer.iterations,
                                       info_string='test attention ')
                        display_mel(pred[kind]['mel'], step=transformers[kind].optimizer.iterations,
                                       info_string='test mel {}'.format(i))
                        display_mel(mel_target, step=transformers['mel_to_mel'].optimizer.iterations,
                            info_string='target mel {}'.format(i))

        print(f'\nbatch {batch_count}')
        for kind in kinds:
            with summary_writers[kind].as_default():
                tf.summary.scalar('loss', output[kind]['loss'], step=transformers[kind].optimizer.iterations)
            print(f'{kind} mean loss: {sum(losses[kind]) / len(losses[kind])}')
        
        if batch_count % text_freq == 0:
            pred = {}
            test_val = {}
            for i in range(0, 3):
                test_val['mel_to_text'] = mel_text_stop_samples[i][0]
                test_val['text_to_text'] = tokenizer.encode(mel_text_stop_samples[i][1])
                decoded_target = tokenizer.decode(test_val['text_to_text'])
                for kind in ['mel_to_text', 'text_to_text']:
                    pred[kind] = transformers[kind].predict(test_val[kind])
                    pred[kind] = tokenizer.decode(pred[kind]['output'])
                    with summary_writers[kind].as_default():
                        tf.summary.text(f'{kind} from validation', f'(pred) {pred[kind]}\n(target) {decoded_target}',
                                        step=transformers[kind].optimizer.iterations)

        batch_count += 1
