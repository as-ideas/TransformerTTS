from typing import List
from abc import ABC, abstractmethod

import torch
import numpy as np

from vocoding.melgan.generator import Generator
from vocoding.hifigan.env import AttrDict
from vocoding.hifigan.models import Generator as GeneratorHiFiGAN


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
    def pretrained(cls) -> 'MelGANPredictor':
        device = get_device()
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://github.com/seungwonpark/melgan/releases/download/v0.3-alpha/nvidia_tacotron2_LJ11_epoch6400.pt',
            map_location=device)
        model = Generator(80, num_layers=[3, 3, 3, 3])
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
    def pretrained(cls) -> 'HiFiGANPredictor':
        device = get_device()
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://github.com/as-ideas/TransformerTTS/releases/download/v1.5/HiFiGAN_model.pt',
            map_location=device)
        h = AttrDict(checkpoint['config'])
        model = GeneratorHiFiGAN(h)
        model.load_state_dict(checkpoint['generator'])
        model.eval()
        model = model.to(device)
        return cls(vocoder_model=model, device=device)