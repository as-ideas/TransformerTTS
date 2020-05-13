# ForwardTransformerTTS: a Text to Speech Transformer
Implementation of a non-autoregressive Transformer based neural network for TTS.<br>
This repo is based on the following papers:
- [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)

Samples of the AutoRegressive models can be found here
https://as-ideas.github.io/TransformerTTS/ <br>
The spectrograms are converted using the pre-trained vocoder model [WaveRNN](https://github.com/fatchord/WaveRNN).<br>
Spectrograms produced with LJSpeech and standard data configuration from this repo are compatible with [WaveRNN](https://github.com/fatchord/WaveRNN).

## Contents
- [Setup](#setup)
- [Training](#training)
    - [Autoregressive](#train_autoregressive_model)
    - [Forward](#train_forward_model)
- [Prediction](#prediction)

## Setup
Install espeak as described [here](https://github.com/bootphon/phonemizer), then
```bash
pip install -r requirements.txt
```
add the project to the paths
```bash
export PYTHONPATH='.'
```
Read the individual scripts for more command line arguments.
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

### Prepare configuration folder
To train on LJSpeech, or if unsure, simply use ```config/standard```.<br>
Alternatively create 3 ```yaml``` configuration files:
 - `autoregressive_config.yaml` contains the settings for creating and training the AutoRegressive model;
 - `forward_config.yaml` contains the settings for creating and training the Forward model;
 - `data_config.yaml` contains the settings processing the data, it's sued by both models.

Note: configurations files are dataset dependent, ```config/standard``` is tuned for LJSpeech v1.1.

### Process dataset
From the root folder run

```bash
python scripts/create_dataset.py \
    --datadir /path/to/dataset/ \
    --targetdir /directory/to/store/spectrograms/ \
    --config /path/to/config/folder/ 
```
for instance:
```bash
python scripts/create_dataset.py \
    --datadir /home/user/LJSpeech-1.1 \
    --targetdir /home/user/data_folder \
    --config config/standard 
```
## Training
### Train Autoregressive Model
From the root folder run
```bash
python scripts/train.py
    --datadir /path/to/spectrograms/
    --logdir /logs/directory/
    --config /path/to/config_folder/
    [--reset_dir | optional flag, deletes all logs and weights from previous sessions]
```
for instance:
```bash
python scripts/train.py
    --datadir /home/user/data_folder
    --logdir /home/user/logs
    --config config/standard
```
### Train Forward Model
#### Compute alignment dataset
In order to train the model to predict the phoneme durations, first use the autoregressive model to create the durations dataset
```bash
python scripts/extract_durations.py 
    --datadir /path/to/spectrograms/
    --logdir /logs/directory/from/training/
```
for instance:
```bash
python scripts/extract_durations.py 
    --datadir /home/user/data_folder
    --logdir /home/user/logs/standard
```
this will add an additional folder to the dataset folder containing the new datasets for validation and training of the forward model.
#### Training
```bash
python scripts/train_forward.py
    --datadir /path/to/spectrograms/
    --logdir /logs/directory/
    --config /path/to/config_folder/
```
for instance:
```bash
python scripts/train_forward.py
    --datadir /home/user/data_folder
    --logdir /home/user/logs
    --config config/standard
```

#### Resume training
Simply target an existing log directory with ```--logdir``` to resume training.
#### Monitor training
We log some information that can be visualized with TensorBoard:
```bash
tensorboard --logdir /logs/directory/
```

## Prediction
Predict with either the Forward or AutoRegressive model
```python
import IPython.display as ipd
from utils.config_loader import ConfigLoader
from utils.audio import reconstruct_waveform

# Create a `ConfigLoader` object using a config file and restore a checkpoint or directly load a weights file
config_loader = ConfigLoader('/path/to/config.yaml', model_kind='forward')
model = config_loader.get_model()
model.load_checkpoint('/path/to/checkpoint/forward_weights/', checkpoint_path=None) # optional: specify checkpoint file
# Run predictions
out = model.predict("Please, say something.")

# Convert spectrogram to wav (with griffin lim) and display
wav= reconstruct_waveform(out['mel'].numpy().T, config=config_loader.config)
ipd.display(ipd.Audio(wav, rate=config_loader.config['sampling_rate']))
```

## Maintainers
* Francesco Cardinale, github: [cfrancesco](https://github.com/cfrancesco)

## Special thanks
[WaveRNN](https://github.com/fatchord/WaveRNN): we took the data processing from here and use their vocoder to produce the samples. <br>
[Erogol](https://github.com/erogol): for the lively exchange on TTS topics. <br>

## Copyright
See [LICENSE](LICENSE) for details.
