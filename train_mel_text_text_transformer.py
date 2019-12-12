import os

import tensorflow as tf
import numpy as np

from losses import masked_crossentropy
from model.models import TextTransformer, MelTextTransformer
from model.transformer_factory import get_components


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

mel_text_train_samples = []
text_train_samples = []
count = 0
with open(metafile, 'r', encoding='utf-8') as f:
    for l in f.readlines():
        l_split = l.split('|')
        text = l_split[-1].strip().lower()
        mel_file = os.path.join(mel_path, l_split[0] + '.npy')
        mel = np.load(mel_file)
        mel_text_train_samples.append((mel, text))
        text_train_samples.append(text)
        count += 1
        if count > 10000:
            break

alphabet = set()
for m, t in mel_text_train_samples:
    alphabet.update(list(t))
tokenizer = TestTokenizer(alphabet=sorted(list(alphabet)))
start_tok, end_tok = tokenizer.start_token_index, tokenizer.end_token_index
tokenized_mel_text_train_samples = [(mel, [start_tok] + tokenizer.encode(text) + [end_tok])
                                    for mel, text in mel_text_train_samples]
mel_text_train_gen = lambda: (pair for pair in tokenized_mel_text_train_samples[10:])
mel_text_train_dataset = tf.data.Dataset.from_generator(mel_text_train_gen, output_types=(tf.float64, tf.int64))
mel_text_train_dataset = mel_text_train_dataset.shuffle(10000).padded_batch(4, padded_shapes=([-1, 80], [-1]))
mel_text_train_dataset = mel_text_train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
input_vocab_size = tokenizer.vocab_size
target_vocab_size = tokenizer.vocab_size

# tokenized_text_train_samples = [[start_tok] + tokenizer.encode(text) + [end_tok]
#                                 for text in text_train_samples]
# text_train_gen = lambda: (pair for pair in tokenized_text_train_samples[10:])
# text_train_dataset = tf.data.Dataset.from_generator(text_train_gen, output_types=(tf.float64, tf.int64))
# text_train_dataset = text_train_dataset.shuffle(10000).padded_batch(4, padded_shapes=([-1, 80], [-1]))
# text_train_dataset = text_train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


mel_channels = 80
num_layers = 4
d_model = 256
num_heads = 8
dff = 512
# TODO: differentiate
# DFF_TEXT = 256
dff_prenet = 256
max_position_encoding = 1000
dropout_rate = 0.1

comps = get_components(input_vocab_size=tokenizer.vocab_size,
                       target_vocab_size=tokenizer.vocab_size,
                       mel_channels=mel_channels,
                       num_layers=num_layers,
                       d_model=d_model,
                       num_heads=num_heads,
                       dff=dff,
                       dff_prenet=dff_prenet,
                       max_position_encoding=max_position_encoding,
                       postnet_conv_filters=1,
                       postnet_conv_layers=1,
                       postnet_kernel_size=1,
                       dropout_rate=dropout_rate)

mel_text_transformer = MelTextTransformer(
    encoder_prenet=comps['speech_encoder_prenet'],
    decoder_prenet=comps['text_decoder_prenet'],
    decoder_postnet=comps['text_decoder_postnet'],
    encoder=comps['speech_encoder'],
    decoder=comps['text_decoder'],
    start_token_index=tokenizer.start_token_index,
    end_token_index=tokenizer.end_token_index,
    mel_channels=mel_channels
)

text_transformer = TextTransformer(
    encoder_prenet=comps['text_encoder_prenet'],
    decoder_prenet=comps['text_decoder_prenet'],
    decoder_postnet=comps['text_decoder_postnet'],
    encoder=comps['text_encoder'],
    decoder=comps['text_decoder'],
    start_token_index=tokenizer.start_token_index,
    end_token_index=tokenizer.end_token_index
)

loss_function = masked_crossentropy
optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
mel_text_transformer.compile(loss=loss_function, optimizer=optimizer)
text_transformer.compile(loss=loss_function, optimizer=optimizer)
losses = []
text_losses = []
batch_count = 0
text_to_text_train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train/text_to_text')
mel_to_text_train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train/mel_to_text')
        
    
for epoch in range(10000):
    for (batch, (inp, tar)) in enumerate(mel_text_train_dataset):
        output = mel_text_transformer.train_step(inp, tar)
        text_output = text_transformer.train_step(tar, tar)
        losses.append(float(output['loss']))
        text_losses.append(float(text_output['loss']))
        with mel_to_text_train_summary_writer.as_default():
            for k in output['attention_weights'].keys():
                for i in range(len(output['attention_weights'][k][0])):
                    tf.summary.image(k + f' head{i}', tf.expand_dims(output['attention_weights'][k][:, i,:,:], -1), step=mel_text_transformer.optimizer.iterations)
        with text_to_text_train_summary_writer.as_default():
            tf.summary.scalar('loss', text_output['loss'], step=text_transformer.optimizer.iterations)
            for k in text_output['attention_weights'].keys():
                for i in range(len(text_output['attention_weights'][k][0])):
                    tf.summary.image(k + f' head{i}', tf.expand_dims(text_output['attention_weights'][k][:, i,:,:], -1), step=text_transformer.optimizer.iterations)
        print('\nbatch count: {}, mean loss: {}'.format(
            batch_count, sum(losses) / len(losses)))
        if batch_count % 200 == 20:
            print('\n>>> Training samples')
            for i in range(10, 13):
                pred = mel_text_transformer.predict(mel_text_train_samples[i][0])
                pred = tokenizer.decode(pred['output'].numpy())
                test_sentence = tokenizer.encode(mel_text_train_samples[i][1])
                text_pred = text_transformer.predict(test_sentence)
                text_pred = tokenizer.decode(text_pred['output'])
                print('(target) {}\n(mel to text) {}\n(text to text) {}\n'.format(
                    mel_text_train_samples[i][1], pred, text_pred))
            print('\n>>> Validation samples')
            for i in range(3):
                pred = mel_text_transformer.predict(mel_text_train_samples[i][0])
                pred = tokenizer.decode(pred['output'])
                text_pred = text_transformer.predict(test_sentence)
                text_pred = tokenizer.decode(text_pred['output'])
                print('(target) {}\n(mel to text) {}\n(text to text) {}\n'.format(
                    mel_text_train_samples[i][1], pred, text_pred))
        batch_count += 1
