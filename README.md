<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/as-ideas/TransformerTTS/dev_autoregressive_only/docs/transformertts_logo.png" width="400"/>
    <br>
<p>

<h3 align="center">
<p>A Text-to-Speech Transformer in TensorFlow 2.0
</h3>

Implementation of an autoregressive Transformer based neural network for Text-to-Speech (TTS). <br>
This repo is based on the following paper:
- [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)

The spectrograms are converted using the pre-trained vocoder model of [WaveRNN](https://github.com/fatchord/WaveRNN).<br>
Spectrograms produced with LJSpeech and standard data configuration from this repo are compatible with [WaveRNN](https://github.com/fatchord/WaveRNN).

## 🔈 Samples

[Can be found here.](https://as-ideas.github.io/TransformerTTS/)

## 📖 Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Prediction](#prediction)

## Installation

Make sure you have:

* Python >= 3.6

Install espeak as phonemizer backend (for macOS use brew):
```
sudo apt-get install espeak
```

Then install the rest with pip:
```
pip install -r requirements.txt
```

Read the individual scripts for more command line arguments.

## Dataset
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
``` wav_file_name|transcription ```

#### Process dataset
```bash
python create_dataset.py --config /path/to/config/folder/
```

##### Configure dataset
* To train on LJSpeech, or if unsure, simply use ```config/standard```
* **EDIT PATHS**: in `data_config.yaml` edit the paths to point at the desired folders if using a custom dataset

## Training
```bash
python train.py --config /path/to/config_folder/
```

#### Training & Model configuration
- Training and model settings can be configured in `model_config.yaml`

#### Resume or restart training
- To resume training simply use the same configuration files AND `--session_name` flag, if any
- To restart training, delete the weights and/or the logs from the logs folder with the training flag `--reset_dir` (both) or `--reset_logs`, `--reset_weights`

#### Monitor training
We log some information that can be visualized with TensorBoard:
```bash
tensorboard --logdir /logs/directory/
```

## Prediction
```python
from utils.config_manager import ConfigManager
from utils.audio import reconstruct_waveform

config_loader = ConfigManager('/path/to/config/')
model = config_loader.get_model()
model.load_checkpoint('/path/to/checkpoint/model_weights/', checkpoint_path=None) # optional: specify checkpoint file
out = model.predict('Please, say something.')

# Convert spectrogram to wav (with griffin lim)
wav = reconstruct_waveform(out['mel'].numpy().T, config=config_loader.config)
```

## Maintainers
* Francesco Cardinale, github: [cfrancesco](https://github.com/cfrancesco)

## Special thanks
[WaveRNN](https://github.com/fatchord/WaveRNN): we took the data processing from here and use their vocoder to produce the samples. <br>
[Erogol](https://github.com/erogol): for the lively exchange on TTS topics. <br>

## Copyright
See [LICENSE](LICENSE) for details.
