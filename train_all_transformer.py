import os

import tensorflow as tf
import numpy as np

from losses import masked_crossentropy, masked_mean_squared_error
from model.transformer_factory import new_everything


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


mel_path = '/Users/fcardina/forge/text-to-speech/data/LJSpeech-1.1/train_mels/mels'
metafile = '/Users/fcardina/forge/text-to-speech/data/LJSpeech-1.1/train_mels/train_metafile.txt'

mel_channels = 80
start_vec = np.ones((1, mel_channels)) * -3
end_vec = np.ones((1, mel_channels))
def get_norm_mel(mel_path):
    mel = np.load(mel_path)
    norm_mel = np.log(mel.clip(1e-5))
    norm_mel = np.concatenate([start_vec, norm_mel, end_vec])
    return norm_mel

mel_text_train_samples = []
text_train_samples = []
count = 0
with open(metafile, 'r', encoding='utf-8') as f:
    for l in f.readlines():
        l_split = l.split('|')
        text = l_split[-1].strip().lower()
        mel_file = os.path.join(mel_path, l_split[0] + '.npy')
        norm_mel = get_norm_mel(mel_file)
        stop_probs = np.ones(norm_mel.shape[0], dtype=np.int64)
        stop_probs[-1] = 2
        mel_text_train_samples.append((norm_mel, text, stop_probs))
        text_train_samples.append(text)
        count += 1
        if count > 20:
            break

alphabet = set()
for m, t, s in mel_text_train_samples:
    alphabet.update(list(t))
tokenizer = TestTokenizer(alphabet=sorted(list(alphabet)))
start_tok, end_tok = tokenizer.start_token_index, tokenizer.end_token_index
tokenized_mel_text_train_samples = [(mel, [start_tok] + tokenizer.encode(text) + [end_tok], stop_prob)
                                    for mel, text, stop_prob in mel_text_train_samples]
mel_text_train_gen = lambda: (pair for pair in tokenized_mel_text_train_samples[10:])
mel_text_train_dataset = tf.data.Dataset.from_generator(mel_text_train_gen, output_types=(tf.float64, tf.int64, tf.int64))
mel_text_train_dataset = mel_text_train_dataset.shuffle(10000).padded_batch(4, padded_shapes=([-1, 80], [-1], [-1]))
mel_text_train_dataset = mel_text_train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
input_vocab_size = tokenizer.vocab_size
target_vocab_size = tokenizer.vocab_size


num_layers = 4
d_model = 256
num_heads = 8
dff = 512
# TODO: differentiate
# DFF_TEXT = 256
dff_prenet = 256
max_position_encoding = 1000
dropout_rate = 0.1

stop_prob_index = 2
start_token_index = tokenizer.start_token_index
end_token_index = tokenizer.end_token_index
postnet_conv_filters = 256
postnet_conv_layers = 5
postnet_kernel_size = 5
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
                                    optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9))
transformers['text_to_text'].compile(loss=masked_crossentropy,
                                     optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9))
transformers['mel_to_mel'].compile(loss=[masked_mean_squared_error,
                                         masked_crossentropy,
                                         masked_mean_squared_error],
                                   loss_weights=loss_coeffs,
                                   optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9))
transformers['text_to_mel'].compile(loss=[masked_mean_squared_error,
                                          masked_crossentropy,
                                          masked_mean_squared_error],
                                    loss_weights=loss_coeffs,
                                    optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9))
batch_count = 0
losses = {}
summary_writers = {}
image_freq = 20
kinds = ['text_to_text', 'mel_to_mel', 'text_to_mel', 'mel_to_text']


def plot_attention(outputs, step):
    for k in outputs[kind]['attention_weights'].keys():
        for i in range(len(outputs['attention_weights'][k][0])):
            tf.summary.image(k + f' head{i}', tf.expand_dims(outputs['attention_weights'][k][:, i, :, :], -1),
                             step=step)


for kind in kinds:
    summary_writers[kind] = tf.summary.create_file_writer(f'/tmp/summaries/train/{kind}')
    losses[kind] = []

for epoch in range(10000):
    for (batch, (mel, text, stop)) in enumerate(mel_text_train_dataset):
        output = {}
        output['text_to_text'] = transformers['text_to_text'].train_step(text, text)
        output['mel_to_mel'] = transformers['mel_to_mel'].train_step(mel, mel, stop)
        output['text_to_mel'] = transformers['text_to_mel'].train_step(text, mel, stop)
        output['mel_to_text'] = transformers['mel_to_text'].train_step(mel, text)
        for kind in kinds:
            losses[kind].append(float(output[kind]['loss']))
        
        if batch % image_freq == image_freq - 1:
            for kind in kinds:
                with summary_writers[kind].as_default():
                    plot_attention(output[kind], step=transformers[kind].optimizer.iterations)
        
        print(f'\nbatch {batch_count}')
        for kind in kinds:
            print(f'{kind} mean loss: {sum(losses[kind]) / len(losses[kind])}')
        
        if batch_count % 200 == 20:
            print('\n>>> Training samples')
            for i in range(10, 13):
                test_mel = mel_text_train_samples[i][0]
                test_sentence = tokenizer.encode(mel_text_train_samples[i][1])
                pred = transformers['mel_to_text'].predict(test_mel)
                pred = tokenizer.decode(pred['output'].numpy())
                text_pred = transformers['text_to_text'].predict(test_sentence)
                text_pred = tokenizer.decode(text_pred['output'])
                print('(target) {}\n(mel to text) {}\n(text to text) {}\n'.format(
                    mel_text_train_samples[i][1], pred, text_pred))
            print('\n>>> Validation samples')
            for i in range(3):
                test_mel = mel_text_train_samples[i][0]
                test_sentence = tokenizer.encode(mel_text_train_samples[i][1])
                pred = transformers['mel_to_text'].predict(test_mel)
                pred = tokenizer.decode(pred['output'].numpy())
                text_pred = transformers['text_to_text'].predict(test_sentence)
                text_pred = tokenizer.decode(text_pred['output'])
                print('(target) {}\n(mel to text) {}\n(text to text) {}\n'.format(
                    mel_text_train_samples[i][1], pred, text_pred))
        batch_count += 1
