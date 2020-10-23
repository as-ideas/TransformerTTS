import tensorflow as tf


def mel_padding_mask(mel_batch, padding_value=0):
    return 1.0 - tf.cast(mel_batch == padding_value, tf.float32)


def mel_lengths(mel_batch, padding_value=0):
    mask = mel_padding_mask(mel_batch, padding_value=padding_value)
    mel_channels = tf.shape(mel_batch)[-1]
    sum_tot = tf.cast(mel_channels, tf.float32) * padding_value
    idxs = tf.cast(tf.reduce_sum(mask, axis=-1) != sum_tot, tf.int32)
    return tf.reduce_sum(idxs, axis=-1)

def phoneme_lengths(phonemes, phoneme_padding=0):
    return tf.reduce_sum(tf.cast(phonemes != phoneme_padding, tf.int32), axis=-1)


if __name__ == '__main__':
    from preprocessing.metadata_readers import get_preprocessor_by_name
    from preprocessing.datasets import DataReader, AutoregressivePreprocessor, TextMelDataset
    from preprocessing.text.tokenizer import Tokenizer
    from preprocessing.text.symbols import all_phonemes
    from pathlib import Path
    
    ljspeech_folder = '/Volumes/data/datasets/LJSpeech-1.1'
    metadata_path = '/Volumes/data/datasets/LJSpeech-1.1/phonemized_metadata.txt'
    metadata_reader = get_preprocessor_by_name('ljspeech')
    data_reader = DataReader(data_directory=ljspeech_folder, metadata_path=metadata_path,
                             metadata_reading_function=metadata_reader, scan_wavs=True)
    mel_dir = Path('/Volumes/data/datasets/LJSpeech-1.1/mels')
    
    tokenizer = Tokenizer()
    preprocessor = AutoregressivePreprocessor(mel_channels=80,
                                              mel_start_value=.5,
                                              mel_end_value=-.5,
                                              tokenizer=tokenizer)
    dataset_creator = TextMelDataset(data_reader=data_reader,
                                     preprocessor=preprocessor,
                                     mel_directory=mel_dir)
    dataset = dataset_creator.get_dataset(shuffle=True, drop_remainder=False,
                                          bucket_batch_sizes=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1])
    for i in range(10):
        mel, text, stop, fname = dataset.next_batch()
        pass
