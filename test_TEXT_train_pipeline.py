import time
import string

import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

train_samples = [
    ('No bug Transformer.', 'Kein Bug Transformator.'),
    ('I am a student.', 'Ich bin ein Student.'),
    ('Put it on.', ' Setz ihn dir auf.'),
    ('Who removed it.', 'Wer hat ihn entfernt.'),
]
# train_samples = [('I am a student.', 'Ich bin ein Student.')] * 2

BUFFER_SIZE = 10
BATCH_SIZE = 2
MAX_LENGTH = 40
num_layers = 2
d_model = 128
dff = 128
num_heads = 4
dropout_rate = 0.0
EPOCHS = 80


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


class TestTokenizer:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet)}
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.start_token = len(self.alphabet)
        self.end_token = len(self.alphabet) + 1
        self.vocab_size = len(self.alphabet) + 2
        self.idx_to_token[self.start_token] = '<START>'
        self.idx_to_token[self.end_token] = '<END>'
    
    def encode(self, sentence):
        return [self.start_token] + [self.token_to_idx[c] for c in sentence] + [self.end_token]
    
    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence])


tokenizer_in = TestTokenizer(alphabet=string.ascii_letters + string.punctuation + string.whitespace)
tokenizer_out = TestTokenizer(alphabet=string.printable)

tokenized_train_samples = [(tokenizer_in.encode(i), tokenizer_out.encode(j)) for i, j in train_samples]
train_gen = lambda: (pair for pair in tokenized_train_samples)
test_gen = lambda: (pair for pair in tokenized_train_samples)
train_examples = tf.data.Dataset.from_generator(train_gen, output_types=(tf.int64, tf.int64))
val_examples = tf.data.Dataset.from_generator(test_gen, output_types=(tf.int64, tf.int64))

train_dataset = train_examples
train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_examples
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))

input_vocab_size = tokenizer_in.vocab_size
target_vocab_size = tokenizer_out.vocab_size

from model.layers import Encoder, Decoder
from model.models import TextTransformer

encoder = Encoder(
    num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, maximum_position_encoding=1000,
    rate=dropout_rate
)

decoder = Decoder(
    num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, maximum_position_encoding=1000,
    rate=dropout_rate
)

transformer = TextTransformer(
    encoder_prenet=tf.keras.layers.Embedding(input_vocab_size, d_model),
    decoder_prenet=tf.keras.layers.Embedding(target_vocab_size, d_model),
    encoder=encoder,
    decoder=decoder,
    vocab_size={'in': tokenizer_in.vocab_size, 'out': tokenizer_out.vocab_size},
)

transformer.compile(loss=loss_object, optimizer=optimizer)

losses = []
for epoch in range(EPOCHS):
    start = time.time()
    for (batch, (inp, tar)) in enumerate(train_dataset):
        gradients, loss, tar_real, predictions = transformer.train_step(inp, tar)
        losses.append(loss)
    
    predicted = tf.cast(tf.argmax(predictions[0], axis=-1), tf.int32).numpy()
    predicted_sentence = transformer.predict(inp[0])['output']
    print('epoch {} loss {}'.format(epoch, float(loss)))
    print('Train output:', tokenizer_out.decode(predicted))
    print('Predicted sentence from ', tokenizer_in.decode(inp[0].numpy()))
    print(tokenizer_out.decode(predicted_sentence))
    print()
