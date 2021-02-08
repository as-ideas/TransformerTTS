from pathlib import Path

import ruamel.yaml
import torch
from vocoding.melgan.generator import Generator
from typing import List
from abc import ABC, abstractmethod
from vocoding.hifigan.env import AttrDict
from vocoding.hifigan.models import Generator as GeneratorHiFiGAN
import json
import numpy as np


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

class WavPredictor(ABC):

    @abstractmethod
    def __call__(self, sentences: List[np.array], **kwargs) -> List[np.array]:
        """ Mel to wav """
        raise NotImplementedError


class MelGANPredictor(WavPredictor):
    
    def __init__(self, vocoder_model: Generator, device: torch.device) -> None:
        super().__init__()
        self.vocoder_model = vocoder_model
        self.device = device
    
    def __call__(self, sentences: List[np.array], **kwargs) -> List[np.array]:
        output = []
        for i, mel in enumerate(sentences):
            mel = mel[np.newaxis, :, :]
            with torch.no_grad():
                t = torch.tensor(mel).to(self.device)
                audio = self.vocoder_model.inference(t)
            output.append(audio.numpy())
        return output
    
    @classmethod
    def from_folder(cls, folder_path: str) -> 'MelGANPredictor':
        device = get_device()
        folder_path = Path(folder_path)
        with open(str(folder_path / 'config.yaml'), 'rb') as data_yaml:
            config = ruamel.yaml.YAML().load(data_yaml)
            model = Generator(80, num_layers=config['num_layers'])
        checkpoint = torch.load(str(folder_path / 'model.pt'), map_location=device)
        model.load_state_dict(checkpoint['model_g'])
        model.eval(inference=True)
        model = model.to(device)
        return cls(vocoder_model=model, device=device)


class HiFiGANPredictor(WavPredictor):
    
    def __init__(self, vocoder_model: Generator, device: torch.device) -> None:
        super().__init__()
        self.vocoder_model = vocoder_model
        self.device = device
    
    def __call__(self, sentences: List[np.array], **kwargs) -> List[np.array]:
        output = []
        for i, mel in enumerate(sentences):
            mel = mel[np.newaxis, :, :]
            with torch.no_grad():
                t = torch.tensor(mel).to(self.device)
                audio = self.vocoder_model(t)
                audio = audio.squeeze()
                MAX_WAV_VALUE = 32768.0
                audio = audio * MAX_WAV_VALUE
                audio = audio.cpu().numpy().astype('int16')
            output.append(audio)
        return output
    
    @classmethod
    def from_folder(cls, folder_path: str) -> 'HiFiGANPredictor':
        device = get_device()
        folder_path = Path(folder_path)
        with open(str(folder_path / 'config.json')) as f:
            data = f.read()
        config = json.loads(data)
        h = AttrDict(config)
        model = GeneratorHiFiGAN(h)
        checkpoint = torch.load(str(folder_path / 'model.pt'), map_location=device)
        model.load_state_dict(checkpoint['generator'])
        model.eval()
        model = model.to(device)
        return cls(vocoder_model=model, device=device)