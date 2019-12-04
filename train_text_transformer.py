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

train_samples = train_samples[:100]

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
    d_model=16,
    num_heads=1,
    dff=128,
    maximum_position_encoding=1000,
    rate=0,
)

decoder = Decoder(
    num_layers=1,
    d_model=16,
    num_heads=1,
    dff=128,
    maximum_position_encoding=1000,
    rate=0,
)

transformer = TextTransformer(
    encoder_prenet=tf.keras.layers.Embedding(input_vocab_size, 16),
    decoder_prenet=tf.keras.layers.Embedding(target_vocab_size, 16),
    encoder=encoder,
    decoder=decoder,
    vocab_size={'in': input_vocab_size, 'out': target_vocab_size}
)

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
transformer.compile(loss=loss_function, optimizer=optimizer)

losses = []
batch_count = 0
for epoch in range(100):
    for (batch, (inp, tar)) in enumerate(train_dataset):
        gradients, loss, tar_real, predictions = transformer.train_step(inp, tar)
        losses.append(float(loss))
        pred = transformer.predict(tokenized_train_samples[0][0])
        pred = tokenizer.decode(pred['output'])
        print('\nbatch count: {}, loss: {}\n(target) {}\n(pred) {}'.format(batch_count, loss, train_samples[0][0], pred))
        batch_count += 1