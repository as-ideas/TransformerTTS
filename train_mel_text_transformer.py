import tensorflow as tf
import os
import numpy as np
from model.layers import Encoder, Decoder
from model.models import TextTransformer
from losses import masked_crossentropy
from model.transformer_factory import new_text_transformer, new_mel_text_transformer


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


mel_path = '/tmp/test_mels/mels'
metafile = '/tmp/test_mels/train_metafile.txt'

train_samples = []
count = 0
with open(metafile, 'r', encoding='utf-8') as f:
    for l in f.readlines():
        l_split = l.split('|')
        text = l_split[-1].strip().lower()
        mel_file = os.path.join(mel_path, l_split[0] + '.npy')
        mel = np.load(mel_file)
        train_samples.append((mel, text))
        count += 1
        if count > 10000:
            break


alphabet = set()
for m, t in train_samples:
    alphabet.update(list(t))
tokenizer = TestTokenizer(alphabet=sorted(list(alphabet)))
start_tok, end_tok = tokenizer.start_token_index, tokenizer.end_token_index
tokenized_train_samples = [(mel, [start_tok] + tokenizer.encode(text) + [end_tok])
                           for mel, text in train_samples]
train_gen = lambda: (pair for pair in tokenized_train_samples[10:])
train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float64, tf.int64))
train_dataset = train_dataset.shuffle(10000).padded_batch(4, padded_shapes=([-1, 80], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
input_vocab_size = tokenizer.vocab_size
target_vocab_size = tokenizer.vocab_size

mel_text_transformer = new_mel_text_transformer(
    start_token_index=tokenizer.start_token_index,
    end_token_index=tokenizer.end_token_index,
    target_vocab_size=tokenizer.vocab_size,
    mel_channels=80,
    num_layers=4,
    d_model=256,
    num_heads=8,
    dff=512,
    dff_prenet=256,
    max_position_encoding=1000,
    dropout_rate=0.1,
)

loss_function = masked_crossentropy
optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
mel_text_transformer.compile(loss=loss_function, optimizer=optimizer)
losses = []
batch_count = 0
for epoch in range(10000):
    for (batch, (inp, tar)) in enumerate(train_dataset):
        gradients, loss, tar_real, predictions = mel_text_transformer.train_step(inp, tar)
        losses.append(float(loss))
        print('\nbatch count: {}, mean loss: {}'.format(
            batch_count, sum(losses)/len(losses)))
        if batch_count % 1000 == 0:
            for i in range(10, 13):
                pred = mel_text_transformer.predict(train_samples[i][0])
                pred = tokenizer.decode(pred['output'])
                print('\n(target) {}\n(pred) {}'.format(
                    train_samples[i][1], pred))
            for i in range(3):
                pred = mel_text_transformer.predict(train_samples[i][0])
                pred = tokenizer.decode(pred['output'])
                print('\n(target) {}\n(pred) {}'.format(
                    train_samples[i][1], pred))
        batch_count += 1