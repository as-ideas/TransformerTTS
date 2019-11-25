import tensorflow as tf
import time
import numpy as np
import string
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
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
transformer.prepare_for_training(loss_object)

losses = []
for epoch in range(EPOCHS):
    start = time.time()
    for (batch, (inp, tar)) in enumerate(train_dataset):
        gradients, loss, tar_real, predictions = transformer.train_step(inp, tar)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        losses.append(loss)
        print('loss {}'.format(loss))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


out_dict = evaluate(TEST_SENTENCE)

losses = [x.numpy() for x in losses]
fix_losses = np.array([4.993565, 4.983845, 4.971134, 4.944314, 4.961081, 4.991240, 4.919794, 4.956748, 4.962680])
np.testing.assert_almost_equal(losses, fix_losses, decimal=6, err_msg='You fucked up.', verbose=True)


np.testing.assert_almost_equal(
    np.sum(out_dict['logits'].numpy()), -0.521198093891, decimal=8, err_msg='You fucked up.', verbose=True
)

print('All good.')
