import tensorflow as tf
import time
import numpy as np
import string
from transformer.model import Transformer

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
    def __init__(self):
        self.alphabet = string.printable
        self.vocab_size = len(self.alphabet)
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet)}
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}

    def encode(self, sentence):
        return [self.token_to_idx[c] for c in sentence]

    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence])


def evaluate(inp_sentence):
    start_token = [char_tokenizer.vocab_size]
    end_token = [char_tokenizer.vocab_size + 1]
    encoded_inp_sentence = start_token + char_tokenizer.encode(inp_sentence) + end_token
    out_dict = transformer.predict(encoded_inp_sentence, MAX_LENGTH=MAX_LENGTH)
    return out_dict


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


char_tokenizer = TestTokenizer()
tokenized_train_samples = [(char_tokenizer.encode(i), char_tokenizer.encode(j)) for i, j in train_samples]
train_gen = lambda: (pair for pair in tokenized_train_samples)
train_examples = tf.data.Dataset.from_generator(train_gen, output_types=(tf.int64, tf.int64))
train_dataset = train_examples.filter(filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

input_vocab_size = char_tokenizer.vocab_size + 2
target_vocab_size = char_tokenizer.vocab_size + 2
transformer = Transformer(
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

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

out_dict = evaluate(TEST_SENTENCE)

losses = [x.numpy() for x in losses]
fix_losses = np.array(
    [4.9885129929, 4.9872484207, 4.9718441963, 4.9400744438, 4.9806327820, 4.9812960625, 4.9257073402, 4.9499497414, 4.9582405090]
)
np.testing.assert_almost_equal(losses, fix_losses, decimal=6, err_msg='You fucked up.', verbose=True)

fix_predictions = np.array(
    [
        0.9125083089,
        -0.2009560168,
        -0.6837264299,
        0.4616462886,
        0.8634129763,
        0.4953508973,
        -0.0735140070,
        -0.6772767901,
        -0.2896357477,
        0.3441579640,
        -0.9505410194,
        -0.2890227437,
        -0.5683166385,
        0.4658229053,
        -0.6880496144,
        0.1015641764,
        -0.4217500389,
        0.2377177477,
        -1.9584358931,
        0.0327272788,
        -0.2444504946,
        0.6142373681,
        -0.2861164510,
        0.2497710735,
        -0.8408367038,
        0.7881574035,
        0.0804156214,
        0.4042626023,
        -0.0150739681,
        0.3187904656,
        -0.4316155016,
        -0.0001461577,
        1.0770248175,
        -0.3024110198,
        0.2539608181,
        0.2527403533,
        -0.3426395655,
        -0.0564526282,
        0.6641074419,
        -0.1961371303,
        -0.5984748006,
        0.1248449907,
        -0.0138634890,
        0.9484200478,
        -0.1657840908,
        -0.7806041241,
        0.7415317297,
        1.2958395481,
        0.6403365731,
        0.4441596568,
        -0.3909725249,
        0.1631689221,
        0.4118253291,
        -0.0320768729,
        0.4760981202,
        -0.2748389542,
        0.2024041116,
        -0.6276949048,
        0.0172423907,
        0.1451888382,
        0.6260839701,
        -0.4427490234,
        -0.8862972260,
        0.0716873184,
        -0.2937331796,
        -1.0517262220,
        0.3452790976,
        -0.4368462861,
        0.7663186193,
        0.4289672673,
        -0.2530202568,
        0.3723466694,
        -0.3176276684,
        -0.1508234292,
        -0.5240167379,
        0.1873067468,
        -0.7936846614,
        0.2141939849,
        0.6852422357,
        -0.5358601213,
        -0.4303381145,
        -0.0300488509,
        0.3897665441,
        0.1209539175,
        -0.2341972142,
        0.4075141549,
        0.3746150136,
        -0.3173985183,
        0.2869723737,
        -0.3450300097,
        -0.2500200570,
        -0.0232751537,
        0.0127873495,
        0.4678322971,
        -0.1292736381,
        0.3396827877,
        -0.4936395586,
        -0.2901942134,
        0.5148264766,
        0.1272658408,
        -0.3138936162,
        -0.5919730067,
    ]
)

np.testing.assert_almost_equal(
    out_dict['logits'].numpy().flatten(), fix_predictions, decimal=6, err_msg='You fucked up.', verbose=True
)

print('All good.')
