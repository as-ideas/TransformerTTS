from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).absolute().parent
sys.path.append(SCRIPT_DIR.parent.as_posix())

control_values_path = Path(SCRIPT_DIR) / 'TextTransformer_control_values.pkl'
if not control_values_path.exists():
    print('MISSING CONTROL VALUES')
    print(f'First create control values by running \n{SCRIPT_DIR}/get_control_value.py')
    exit()

import pickle
import string
import time
import numpy as np
import tensorflow as tf
from src.models import TextTransformer

np.random.seed(42)
tf.random.set_seed(42)

train_samples = [('I am a student.', 'Ich bin ein Student.')] * 2
test_samples = [('I am a student.', 'Ich bin ein Student.')] * 2
TEST_SENTENCE = 'Some sentence to be checked. right now'
BUFFER_SIZE = 10
BATCH_SIZE = 2
MAX_LENGTH = 40
num_layers = 2
d_model = 16
dff = 32
num_heads = 2
dropout_rate = 0.1
EPOCHS = 9


class TestTokenizer:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.vocab_size = len(self.alphabet)
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet)}
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}

    def encode(self, sentence):
        return [self.token_to_idx[c] for c in sentence]

    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence])


def evaluate(inp_sentence):
    start_token = [tokenizer_in.vocab_size]
    end_token = [tokenizer_in.vocab_size + 1]
    encoded_inp_sentence = start_token + tokenizer_in.encode(inp_sentence) + end_token
    out_dict = transformer.predict(encoded_inp_sentence, MAX_LENGTH=MAX_LENGTH)
    return out_dict


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


tokenizer_in = TestTokenizer(alphabet=string.ascii_letters + string.punctuation + string.whitespace)
tokenizer_out = TestTokenizer(alphabet=string.printable)
tokenized_train_samples = [(tokenizer_in.encode(i), tokenizer_out.encode(j)) for i, j in train_samples]
train_gen = lambda: (pair for pair in tokenized_train_samples)
train_examples = tf.data.Dataset.from_generator(train_gen, output_types=(tf.int64, tf.int64))
train_dataset = train_examples.filter(filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

input_vocab_size = tokenizer_in.vocab_size + 2
target_vocab_size = tokenizer_out.vocab_size + 2
transformer = TextTransformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    vocab_size={'in': input_vocab_size, 'out': target_vocab_size},
    pe_input=input_vocab_size,
    pe_target=target_vocab_size,
    rate=dropout_rate,
)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
transformer.compile(loss=loss_function, optimizer=optimizer)

losses = []
for epoch in range(EPOCHS):
    start = time.time()
    for (batch, (inp, tar)) in enumerate(train_dataset):
        gradients, loss, tar_real, predictions = transformer.train_step(inp, tar)
        # optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        losses.append(loss)

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

losses = np.array([x.numpy() for x in losses])
out_dict = evaluate(TEST_SENTENCE)

test_values = {'losses': losses, 'logits_sum': np.sum(out_dict['logits'].numpy())}

np.set_printoptions(formatter={'float': lambda x: "{0:0.9f}".format(x)})

print('losses:')
for l in losses:
    print(f'{l}')
print(f'logits sum:{test_values["logits_sum"]}')


cv = pickle.load(open(control_values_path.as_posix(), 'rb'))
equal = {}
consistent = True
for k in ['losses', 'logits_sum']:
    equal[k] = np.allclose(cv[k], test_values[k])
    print(f'{k} equal: {equal[k]}')
    if not equal[k]:
        consistent = False
if not consistent:
    print('Some values are inconsistent. Check your settings and/or environment.')
else:
    print('All good.')
