<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/as-ideas/TransformerTTS/master/docs/transformer_logo.png" width="400"/>
    <br>
</p>

<h2 align="center">
<p>A Text-to-Speech Transformer in TensorFlow 2</p>
</h2>

Implementation of a non-autoregressive Transformer based neural network for Text-to-Speech (TTS). <br>
This repo is based, among others, on the following papers:
- [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- [FastPitch: Parallel Text-to-speech with Pitch Prediction](https://fastpitch.github.io/)

Our pre-trained LJSpeech model is compatible with the pre-trained vocoders:
- [MelGAN](https://github.com/seungwonpark/melgan)
- [HiFiGAN](https://github.com/jik876/hifi-gan)

(older versions are available also for [WaveRNN](https://github.com/fatchord/WaveRNN))

For quick inference with these vocoders, checkout the [Vocoding branch](https://github.com/as-ideas/TransformerTTS/tree/vocoding)

#### Non-Autoregressive
Being non-autoregressive, this Transformer model is:
- Robust: No repeats and failed attention modes for challenging sentences.
- Fast: With no autoregression, predictions take a fraction of the time.
- Controllable: It is possible to control the speed and pitch of the generated utterance.

## 🔈 Samples

[Can be found here.](https://as-ideas.github.io/TransformerTTS/)

These samples' spectrograms are converted using the pre-trained [MelGAN](https://github.com/seungwonpark/melgan) vocoder.<br>


Try it out on Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as-ideas/TransformerTTS/blob/main/notebooks/synthesize_forward_melgan.ipynb)

## Updates
- 06/20: Added normalisation and pre-trained models compatible with the faster [MelGAN](https://github.com/seungwonpark/melgan) vocoder.
- 11/20: Added pitch prediction. Autoregressive model is now specialized as an Aligner and Forward is now the only TTS model. Changed models architectures. Discontinued WaveRNN support. Improved duration extraction with Dijkstra algorithm.
- 03/20: Vocoding branch.

## 📖 Contents
- [Installation](#installation)
- [API](#pre-trained-ljspeech-api)
- [Dataset](#dataset)
- [Training](#training)
    - [Aligner](#train-aligner-model)
    - [TTS](#train-tts-model)
- [Prediction](#prediction)
- [Model Weights](#model-weights)

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

## Pre-Trained LJSpeech API
Use our pre-trained model (with Griffin-Lim) from command line with
```commandline
python predict_tts.py -t "Please, say something."
```
Or in a python script
```python
from data.audio import Audio
from model.factory import tts_ljspeech

model = tts_ljspeech()
audio = Audio.from_config(model.config)
out = model.predict('Please, say something.')

# Convert spectrogram to wav (with griffin lim)
wav = audio.reconstruct_waveform(out['mel'].numpy().T)
```

You can specify the model step with the `--step` flag (CL) or `step` parameter (script).<br>
Steps from 60000 to 100000 are available at a frequency of 5K steps (60000, 65000, ..., 95000, 100000).

<b>IMPORTANT:</b> make sure to checkout the correct repository version to use the API.<br>
Currently 493be6345341af0df3ae829de79c2793c9afd0ec

## Dataset
You can directly use [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) to create the training dataset.

#### Configuration
* If training on LJSpeech, or if unsure, simply use ```config/session_paths.yaml``` to create [MelGAN](https://github.com/seungwonpark/melgan) or [HiFiGAN](https://github.com/jik876/hifi-gan) compatible models
    * swap ```data_config.yaml``` for ```data_config_wavernn.yaml``` to create models compatible with [WaveRNN](https://github.com/fatchord/WaveRNN) 
* **EDIT PATHS**: in `config/session_paths.yaml` edit the paths to point at your dataset and log folders

#### Custom dataset
Prepare a folder containing your metadata and wav files, for instance
```
|- dataset_folder/
|   |- metadata.csv
|   |- wavs/
|       |- file1.wav
|       |- ...
```
if `metadata.csv` has the following format
``` wav_file_name|transcription ```
you can use the ljspeech preprocessor in ```data/metadata_readers.py```, otherwise add your own under the same file.

Make sure that:
 -  the metadata reader function name is the same as ```data_name``` field in ```session_paths.yaml```.
 -  the metadata file (can be anything) is specified under ```metadata_path``` in ```session_paths.yaml``` 

## Training
Change the ```--config``` argument based on the configuration of your choice.
### Train Aligner Model
#### Create training dataset
```bash
python create_training_data.py --config config/session_paths.yaml
```
This will populate the training data directory (default `transformer_tts_data.ljspeech`).
#### Training
```bash
python train_aligner.py --config config/session_paths.yaml
```
### Train TTS Model
#### Compute alignment dataset
First use the aligner model to create the durations dataset
```bash
python extract_durations.py --config config/session_paths.yaml
```
this will add the `durations.<session name>` as well as the char-wise pitch folders to the training data directory.
#### Training
```bash
python train_tts.py --config config/session_paths.yaml
```
#### Training & Model configuration
- Training and model settings can be configured in `<model>_config.yaml`

#### Resume or restart training
- To resume training simply use the same configuration files
- To restart training, delete the weights and/or the logs from the logs folder with the training flag `--reset_dir` (both) or `--reset_logs`, `--reset_weights`

#### Monitor training
```bash
tensorboard --logdir /logs/directory/
```

![Tensorboard Demo](https://raw.githubusercontent.com/as-ideas/TransformerTTS/master/docs/tboard_demo.gif)
#### Checkpoint to hdf5 weights \[optional\]
You can convert the checkpoint files to hdf5 model weights by running
```bash
python checkpoints_to_weights.py --config config/session_paths.yaml
```
## Prediction
### With model weights
From command line with
```commandline
python predict_tts.py -t "Please, say something." -p /path/to/weights/
```
Or in a python script
```python
from model.models import ForwardTransformer
from data.audio import Audio
model = ForwardTransformer.load_model('/path/to/weights/')
audio = Audio.from_config(model.config)
out = model.predict('Please, say something.')

# Convert spectrogram to wav (with griffin lim)
wav = audio.reconstruct_waveform(out['mel'].numpy().T)
```

## Model Weights
Access the pre-trained models with the API call.

Old weights
| Model URL | Commit | Vocoder Commit|
|---|---|---|
|[ljspeech_tts_model](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ljspeech_weights_tts.zip)| 0cd7d33 | aca5990 |
|[ljspeech_melgan_forward_model](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_melgan_forward_transformer.zip)| 1c1cb03| aca5990 |
|[ljspeech_melgan_autoregressive_model_v2](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_melgan_autoregressive_transformer.zip)| 1c1cb03| aca5990 |
|[ljspeech_wavernn_forward_model](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_wavernn_forward_transformer.zip)| 1c1cb03| 3595219 |
|[ljspeech_wavernn_autoregressive_model_v2](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_wavernn_autoregressive_transformer.zip)| 1c1cb03| 3595219 |
|[ljspeech_wavernn_forward_model](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_forward_transformer.zip)| d9ccee6| 3595219 |
|[ljspeech_wavernn_autoregressive_model_v2](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_autoregressive_transformer.zip)| d9ccee6| 3595219 |
|[ljspeech_wavernn_autoregressive_model_v1](https://github.com/as-ideas/tts_model_outputs/tree/master/ljspeech_transformertts)| 2f3a1b5| 3595219 |
## Maintainers
* Francesco Cardinale, github: [cfrancesco](https://github.com/cfrancesco)

## Special thanks
[MelGAN](https://github.com/seungwonpark/melgan) and [WaveRNN](https://github.com/fatchord/WaveRNN): data normalization and samples' vocoders are from these repos.

[Erogol](https://github.com/erogol) and the Mozilla TTS team for the lively exchange on the topic.


## Copyright
See [LICENSE](LICENSE) for details.
