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
python scripts/create_dataset.py 
    --metafile /path/to/metadata.csv 
    --wavdir /path/to/wav/directory/
    --targetdir /directory/to/store/spectrograms/
    --config /path/to/config/file.yaml
```
### Run training
From the root folder run
```
python scripts/train.py
    --datadir /path/to/spectrograms/
    --logdir /logs/directory/
    --config /path/to/config_file.yaml
    [--reset_dir | optional flag, deletes all logs and weights from previous sessions]
```
#### Resume training
Simply target an existing log directory with ```--logdir``` to resume training.
#### Monitor training
We log some information that can be visualized with TensorBoard:
```bash
tensorboard --logdir /logs/directory/
```

## Prediction
In a Jupyter notebook
```python
import IPython.display as ipd
from utils.config_loader import ConfigLoader
from utils.audio import reconstruct_waveform

# Create a `ConfigLoader` object using a config file and restore a checkpoint or directly load a weights file
config_loader = ConfigLoader('/path/to/config.yaml')
model = config_loader.get_model()
model.load_checkpoint('/path/to/checkpoint/weights/', checkpoint_path=None) # optional: specify checkpoint file
# Run predictions
out = model.predict("Please, say something.")

# Convert spectrogram to wav (with griffin lim) and display
wav= reconstruct_waveform(out['mel'].numpy().T, config=config_loader.config)
ipd.display(ipd.Audio(wav, rate=config_loader.config['sampling_rate']))
```

## Maintainers

* Francesco Cardinale, github: [cfrancesco](https://github.com/cfrancesco)

## Copyright

See [LICENSE](LICENSE) for details.
