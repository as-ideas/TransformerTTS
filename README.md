# TransformerTTS: a Text to Speech Transformer
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

Read the individual scripts for more command line arguments.
### Dataset
You can directly use [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).
#### Custom dataset
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

### Configuration
To train on LJSpeech, or if unsure, simply use ```config/standard```.<br>
**EDIT PATHS**: in `data_config.yaml` edit the paths to point at the desired folders.<br>
#### Custom configuration
Configurations files are dataset dependent, you will need to tune```config/standard``` if using a different dataset.<br>
The configuration files are:
 - `autoregressive_config.yaml`: autoregressive model training and architecture settings;
 - `forward_config.yaml`: forward model training and architecture settings;
 - `data_config.yaml`: data processing settings, used by both models.

### Process dataset
```bash
python create_dataset.py --config /path/to/config/folder/ 
```
## Training
### Train Autoregressive Model
```bash
python train.py --config /path/to/config_folder/
```
### Train Forward Model
#### Compute alignment dataset
In order to train the model to predict the phoneme durations, first use the autoregressive model to create the durations dataset
```bash
python extract_durations.py --config /path/to/config_folder/
```
this will add an additional folder to the dataset folder containing the new datasets for validation and training of the forward model.
#### Training
```bash
python train_forward.py  --config /path/to/config_folder/
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
from utils.config_manager import ConfigManager
from utils.audio import reconstruct_waveform

config_loader = ConfigManager('/path/to/config/', model_kind='forward')
model = config_loader.get_forward_model()
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
