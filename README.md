# Text to Speech Transformer
Implementation of a Transformer based neural network for text to speech.

## Contents
- [Training](#training)
- [Prediction](#prediction-wip)

## Training
### Prepare dataset folder
Prepare a dataset in the following format:
```
|- dataset_folder/
|   |- metadata.csv
|   |- wav/
|       |- file1.wav
|       |- ...
```
where `metadata.csv` has the following format: 
``` wav_file_name|trascription ```

The dataset [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) is in this format.

### Prepare configuration file
Prepare a ```yaml``` configuration file with architecture, data and training configurations.

If unsure, simply use ```config/standard_config.yaml```.

Note: configurations files are dataset dependent, ```standard_config.yaml``` is tuned for LJSpeech1.1.

### Process dataset
From the root folder run
```
python scripts/preprocess_mels.py 
    --meta_file /path/to/metadata.csv 
    --wav_dir /path/to/wav/directory/
    --target_dir /directory/to/store/spectrograms/
    --config /path/to/config/file.yaml
```
### Run training
From the root folder run
```
python scripts/train.py
    --datadir /path/to/spectrograms/
    --logdir /logs/directory/
    --config /path/to/config_file.yaml
    [--cleardir | optional flag, deletes all logs and weights from previous sessions]
```
#### Resume training
Simply target an existing log directory with ```--logdir``` to resume training.

## Prediction [WIP]
In a Jupiter notebook
```python
import librosa
import ruamel.yaml
import numpy as np
import tensorflow as tf
import IPython.display as ipd
from model.combiner import Combiner
from preprocessing.text_processing import Phonemizer, TextCleaner


# Import a config file
yaml = ruamel.yaml.YAML()
config = yaml.load(open('/path/to/config.yaml', 'rb'))

# Create a `Combiner` object using a config file
combiner = Combiner(config)
combiner.text_mel.set_r(1)
# Do some tensorflow "magic"
try:
    combiner.text_mel.forward([0], output=[0], decoder_prenet_dropout=.5)
except:
    pass

# Restore a checkpoint using the checkpoint manager
ckpt = tf.train.Checkpoint(net=combiner.text_mel)
manager = tf.train.CheckpointManager(ckpt, '/path/to/checkpoint/weights/', max_to_keep=None)
ckpt.restore(manager.latest_checkpoint)

# Prepare sentence for prediction
cleaner = TextCleaner()
phonemizer = Phonemizer(language='en')
sentence = "Plese, say something."
sentence=phonemizer.encode(cleaner.clean(sentence))
enc_sentence = [combiner.tokenizer.start_token_index] + combiner.tokenizer.encode(sentence.lower()) + [combiner.tokenizer.end_token_index]

# Run predictions
out = combiner.text_mel.predict(enc_sentence, max_length=1000, decoder_prenet_dropout=config['dropout_schedule'][-1][-1])

# Convert spectrogram to wav (with griffin lim) and display
stft = librosa.feature.inverse.mel_to_stft(np.exp(out['mel'].numpy().T), sr=22050, n_fft=1024, power=1, fmin=0, fmax=8000) 
wav = librosa.feature.inverse.griffinlim(stft, n_iter=32, hop_length=256, win_length=1024)
ipd.display(ipd.Audio(wav, rate=22050))
```