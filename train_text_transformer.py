import tensorflow as tf

from model.layers import Encoder, Decoder
from model.models import TextTransformer


class TestTokenizer:

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet)}
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.start_token = len(self.alphabet)
        self.end_token = len(self.alphabet) + 1
        self.vocab_size = len(self.alphabet) + 2
        self.idx_to_token[self.start_token] = '<'
        self.idx_to_token[self.end_token] = '>'

    def encode(self, sentence):
        return [self.token_to_idx[c] for c in sentence]

    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence])


metafile = '/tmp/test_mels/train_metafile.txt'

train_samples = []
with open(metafile, 'r', encoding='utf-8') as f:
    for l in f.readlines():
        l_split = l.split('|')
        text = l_split[-1].strip()
        train_samples.append((text, text))

train_samples = train_samples[:-10]
test_samples = train_samples[-10:-1]

alphabet = set()
for t, t in train_samples:
    alphabet.update(list(t))
tokenizer = TestTokenizer(alphabet=sorted(list(alphabet)))
start_tok, end_tok = tokenizer.start_token, tokenizer.end_token
tokenized_train_samples = [([start_tok] + tokenizer.encode(i) + [end_tok],
                            [start_tok] + tokenizer.encode(j) + [end_tok])
                           for i, j in train_samples]
train_gen = lambda: (pair for pair in tokenized_train_samples)
train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.int64, tf.int64))
train_dataset = train_dataset.shuffle(1000).padded_batch(16, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
input_vocab_size = tokenizer.vocab_size
target_vocab_size = tokenizer.vocab_size

encoder = Encoder(
    num_layers=1,
    d_model=128,
    num_heads=2,
    dff=512,
    maximum_position_encoding=1000,
    prenet=tf.keras.layers.Embedding(input_vocab_size, 128),
    rate=0.1,
)

decoder = Decoder(
    num_layers=1,
    d_model=128,
    num_heads=2,
    dff=512,
    maximum_position_encoding=1000,
    prenet=tf.keras.layers.Embedding(target_vocab_size, 128),
    rate=0,
)

transformer = TextTransformer(
    encoder=encoder,
    decoder=decoder,
    start_token=tokenizer.start_token,
    end_token=tokenizer.end_token,
    vocab_size_encoder=input_vocab_size,
    vocab_size_decoder=target_vocab_size,
)
