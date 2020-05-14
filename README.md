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
**EDIT PATHS**: in `data_config.yaml` edit the paths to point at the desired folders.<br>

Alternatively create 3 ```yaml``` configuration files:
 - `autoregressive_config.yaml` contains the settings for creating and training the AutoRegressive model;
 - `forward_config.yaml` contains the settings for creating and training the Forward model;
 - `data_config.yaml` contains the settings processing the data, it's sued by both models.

Note: configurations files are dataset dependent, ```config/standard``` is tuned for LJSpeech v1.1.
### Process dataset
From the root folder run

```bash
python scripts/create_dataset.py --config /path/to/config/folder/ 
```
## Training
### Train Autoregressive Model
From the root folder run
```bash
python scripts/train.py --config /path/to/config_folder/
```
### Train Forward Model
#### Compute alignment dataset
In order to train the model to predict the phoneme durations, first use the autoregressive model to create the durations dataset
```bash
python scripts/extract_durations.py --config /path/to/config_folder/
```
this will add an additional folder to the dataset folder containing the new datasets for validation and training of the forward model.
#### Training
```bash
python scripts/train_forward.py  --config /path/to/config_folder/
```

#### Resume or restart training
To resume training simply use the same configuration files AND `--session_name` flag, if any. <br>
To restart training, delete the weights and/or the logs from the logs folder with the training flag `--reset_dir` (both) or `--reset_logs`, `--reset_weights`. 
#### Monitor training
We log some information that can be visualized with TensorBoard:
```bash
tensorboard --logdir /logs/directory/
```

## Prediction
Predict with either the Forward or AutoRegressive model
```python
from utils.config_loader import ConfigLoader
from utils.audio import reconstruct_waveform

config_loader = ConfigLoader('/path/to/config.yaml', model_kind='forward')
model = config_loader.get_model()
model.load_checkpoint('/path/to/checkpoint/forward_weights/', checkpoint_path=None) # optional: specify checkpoint file
out = model.predict("Please, say something.")

# Convert spectrogram to wav (with griffin lim)
wav= reconstruct_waveform(out['mel'].numpy().T, config=config_loader.config)
```

## Maintainers
* Francesco Cardinale, github: [cfrancesco](https://github.com/cfrancesco)

## Special thanks
[WaveRNN](https://github.com/fatchord/WaveRNN): we took the data processing from here and use their vocoder to produce the samples. <br>
[Erogol](https://github.com/erogol): for the lively exchange on TTS topics. <br>

## Copyright
See [LICENSE](LICENSE) for details.
