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

Note: configurations files are dataset dependent, ```standard_config.yaml``` is tuned for LJSpeech v1.1.

### Process dataset
From the root folder run
```
python scripts/preprocess_mels.py \
    --meta_file /path/to/metadata.csv \
    --wav_dir /path/to/wav/directory/ \
    --target_dir /directory/to/store/spectrograms/ \
    --config /path/to/config_file.yaml
```
### Run training
From the root folder run
```
python scripts/train.py \
    --datadir /path/to/spectrograms/ \
    --logdir /logs/directory/ \
    --config /path/to/config_file.yaml \
    [--cleardir | optional flag, deletes all logs and weights from previous sessions]
```
#### Resume training
Simply target an existing log directory with ```--logdir``` to resume training.

## Prediction
In a Jupyter notebook
```python
import librosa
import numpy as np
import IPython.display as ipd
from utils.config_loader import ConfigLoader

# Create a `ConfigLoader` object using a config file and restore a checkpoint or directly load a weights file
config_loader = ConfigLoader('/path/to/config.yaml')
model = config_loader.get_model()
model.load_weights('weights_new.hdf5')
# model.load_checkpoint('/path/to/checkpoint/weights/', checkpoint_path=None) # optional: specify checkpoint file
# Run predictions
out = model.predict('Please, say something.', encode=True)

# Convert spectrogram to wav (with griffin lim) and display
stft = librosa.feature.inverse.mel_to_stft(np.exp(out['mel'].numpy().T), sr=22050, n_fft=1024, power=1, fmin=0, fmax=8000) 
wav = librosa.feature.inverse.griffinlim(stft, n_iter=32, hop_length=256, win_length=1024)
ipd.display(ipd.Audio(wav, rate=22050))
```

## Maintainers

* Francesco Cardinale, github: [cfrancesco](https://github.com/cfrancesco)

## Copyright

See [LICENSE](LICENSE) for details.