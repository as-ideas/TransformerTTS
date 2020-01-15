import os
import datetime
import argparse
from pathlib import Path

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from model.transformer_factory import Combiner
from losses import masked_crossentropy, masked_mean_squared_error
from utils import buffer_mel

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
parser.add_argument('--mask_prob', dest='MASK_PROB', default=0.3, type=float)
args = parser.parse_args()

mel_path = Path(args.MEL_DIR) / 'mels'
metafile = Path(args.MEL_DIR) / 'train_metafile.txt'

tokenizer_type = 'char'
tokenizer_alphabet = None
speech_encoder_num_layers = 4
speech_decoder_num_layers = 4
text_encoder_num_layers = 4
text_decoder_num_layers = 4
speech_model_dimension = 256
text_model_dimension = 256
speech_encoder_num_heads = 4
speech_decoder_num_heads = 4
text_encoder_num_heads = 4
text_decoder_num_heads = 4
text_encoder_feed_forward_dimension = 512
text_decoder_feed_forward_dimension = 512
speech_encoder_feed_forward_dimension = 512
speech_decoder_feed_forward_dimension = 512
speech_encoder_prenet_dimension = 256
speech_decoder_prenet_dimension = 256
max_position_encoding = 10000
speech_postnet_conv_filters = 256
speech_postnet_conv_layers = 5
speech_postnet_kernel_size = 5
dropout_rate = args.DROPOUT

sr = 22050
N_EPOCHS = args.EPOCHS
N_SAMPLES = args.MAX_SAMPLES
image_freq = args.IMAGE_FREQ
text_freq = args.TEXT_FREQ
mel_channels = args.MEL_CHANNELS


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

combiner = Combiner(
    tokenizer_type='char',
    mel_channels=mel_channels,
    speech_encoder_num_layers=speech_encoder_num_layers,
    speech_decoder_num_layers=speech_decoder_num_layers,
    text_encoder_num_layers=text_encoder_num_layers,
    text_decoder_num_layers=text_decoder_num_layers,
    speech_model_dimension=speech_model_dimension,
    text_model_dimension=text_model_dimension,
    speech_encoder_num_heads=speech_encoder_num_heads,
    speech_decoder_num_heads=speech_decoder_num_heads,
    text_encoder_num_heads=text_encoder_num_heads,
    text_decoder_num_heads=text_decoder_num_heads,
    text_encoder_feed_forward_dimension=text_encoder_feed_forward_dimension,
    text_decoder_feed_forward_dimension=text_decoder_feed_forward_dimension,
    speech_encoder_feed_forward_dimension=speech_encoder_feed_forward_dimension,
    speech_decoder_feed_forward_dimension=speech_decoder_feed_forward_dimension,
    speech_encoder_prenet_dimension=speech_encoder_prenet_dimension,
    speech_decoder_prenet_dimension=speech_decoder_prenet_dimension,
    max_position_encoding=max_position_encoding,
    speech_postnet_conv_filters=speech_postnet_conv_filters,
    speech_postnet_conv_layers=speech_postnet_conv_layers,
    speech_postnet_kernel_size=speech_postnet_kernel_size,
    dropout_rate=dropout_rate,
    mel_start_vec_value=-3,
    mel_end_vec_value=1,
    tokenizer_alphabet=sorted(list(alphabet)),
)
start_tok, end_tok = combiner.tokenizer.start_token_index, combiner.tokenizer.end_token_index
train_list, test_list = train_test_split(mel_text_stop_samples, test_size=100, random_state=42)
tokenized_train_list = [(mel, [start_tok] + combiner.tokenizer.encode(text) + [end_tok], stop_prob)
                        for mel, text, stop_prob in train_list]
tokenized_test_list = [(mel, [start_tok] + combiner.tokenizer.encode(text) + [end_tok], stop_prob)
                       for mel, text, stop_prob in test_list]

train_set_generator = lambda: (item for item in tokenized_train_list)
train_dataset = tf.data.Dataset.from_generator(train_set_generator,
                                               output_types=(tf.float64, tf.int64, tf.int64))
train_dataset = train_dataset.shuffle(1000).padded_batch(
    args.BATCH_SIZE, padded_shapes=([-1, 80], [-1], [-1]), drop_remainder=True)

loss_coeffs = [1.0, 1.0, 1.0]
combiner.transformers['mel_to_text'].compile(loss=masked_crossentropy,
                                             optimizer=tf.keras.optimizers.Adam(args.LEARNING_RATE, beta_1=0.9,
                                                                                beta_2=0.98,
                                                                                epsilon=1e-9))
combiner.transformers['text_to_text'].compile(loss=masked_crossentropy,
                                              optimizer=tf.keras.optimizers.Adam(args.LEARNING_RATE, beta_1=0.9,
                                                                                 beta_2=0.98,
                                                                                 epsilon=1e-9))
combiner.transformers['mel_to_mel'].compile(loss=[masked_mean_squared_error,
                                                  masked_crossentropy,
                                                  masked_mean_squared_error],
                                            loss_weights=loss_coeffs,
                                            optimizer=tf.keras.optimizers.Adam(args.LEARNING_RATE, beta_1=0.9,
                                                                               beta_2=0.98,
                                                                               epsilon=1e-9))
combiner.transformers['text_to_mel'].compile(loss=[masked_mean_squared_error,
                                                   masked_crossentropy,
                                                   masked_mean_squared_error],
                                             loss_weights=loss_coeffs,
                                             optimizer=tf.keras.optimizers.Adam(args.LEARNING_RATE, beta_1=0.9,
                                                                                beta_2=0.98,
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
    losses[kind] = []

weights_paths = f'/tmp/weights/train/{current_time}'


def linear_dropout_schedule(step):
    dout = max(((-0.9 + 0.5) / 20000.) * step + 0.9, .5)
    return tf.cast(dout, tf.float32)


def random_mel_mask(tensor, mask_prob):
    tensor_shape = tf.shape(tensor)
    mask_floats = tf.random.uniform((tensor_shape[0], tensor_shape[1]))
    mask = tf.cast(mask_floats > mask_prob, tf.float64)
    mask = tf.expand_dims(mask, -1)
    mask = tf.broadcast_to(mask, tensor_shape)
    masked_tensor = tensor * mask
    return masked_tensor


def random_text_mask(tensor, mask_prob):
    tensor_shape = tf.shape(tensor)
    mask_floats = tf.random.uniform((tensor_shape[0], tensor_shape[1]))
    mask = tf.cast(mask_floats > mask_prob, tf.int64)
    masked_tensor = tensor * mask
    return masked_tensor


for epoch in range(N_EPOCHS):
    for (batch, (mel, text, stop)) in enumerate(train_dataset):
        decoder_prenet_dropout = linear_dropout_schedule(batch_count)
        output = combiner.train_step(text=text,
                                     mel=mel,
                                     stop=stop,
                                     speech_decoder_prenet_dropout=decoder_prenet_dropout,
                                     mask_prob=args.MASK_PROB)
        with summary_writers['text_to_text'].as_default():
            tf.summary.scalar('dropout', decoder_prenet_dropout,
                              step=combiner.transformers['text_to_text'].optimizer.iterations)
        for kind in kinds:
            losses[kind].append(float(output[kind]['loss']))
        if batch_count % image_freq == 0:
            for kind in kinds:
                with summary_writers[kind].as_default():
                    plot_attention(output[kind], step=combiner.transformers[kind].optimizer.iterations,
                                   info_string='train attention ')
                
                combiner.save_weights(weights_paths, batch_count)
            
            pred = {}
            test_val = {}
            for i in range(0, 3):
                mel_target = test_list[i][0]
                max_pred_len = mel_target.shape[0] + 50
                test_val['text_to_mel'] = combiner.tokenizer.encode(test_list[i][1])
                test_val['mel_to_mel'] = mel_target
                for kind in ['text_to_mel', 'mel_to_mel']:
                    pred[kind] = combiner.transformers[kind].predict(test_val[kind],
                                                                     max_length=max_pred_len,
                                                                     decoder_prenet_dropout=0.5)
                    with summary_writers[kind].as_default():
                        plot_attention(pred[kind], step=combiner.transformers[kind].optimizer.iterations,
                                       info_string='test attention ')
                        display_mel(pred[kind]['mel'], step=combiner.transformers[kind].optimizer.iterations,
                                    info_string='test mel {}'.format(i))
                        display_mel(mel_target, step=combiner.transformers['mel_to_mel'].optimizer.iterations,
                                    info_string='target mel {}'.format(i))
        
        print(f'\nbatch {batch_count}')
        for kind in kinds:
            with summary_writers[kind].as_default():
                tf.summary.scalar('loss', output[kind]['loss'], step=combiner.transformers[kind].optimizer.iterations)
            print(f'{kind} mean loss: {sum(losses[kind]) / len(losses[kind])}')
        
        if batch_count % text_freq == 0:
            pred = {}
            test_val = {}
            for i in range(0, 3):
                test_val['mel_to_text'] = test_list[i][0]
                test_val['text_to_text'] = combiner.tokenizer.encode(test_list[i][1])
                decoded_target = combiner.tokenizer.decode(test_val['text_to_text'])
                for kind in ['mel_to_text', 'text_to_text']:
                    pred[kind] = combiner.transformers[kind].predict(test_val[kind])
                    pred[kind] = combiner.tokenizer.decode(pred[kind]['output'])
                    with summary_writers[kind].as_default():
                        tf.summary.text(f'{kind} from validation', f'(pred) {pred[kind]}\n(target) {decoded_target}',
                                        step=combiner.transformers[kind].optimizer.iterations)
        
        batch_count += 1
