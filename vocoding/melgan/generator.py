"""
CODE ADAPTED FROM
https://github.com/seungwonpark/melgan
"""

import torch
import torch.nn as nn

from .res_stack import ResStack


MAX_WAV_VALUE = 32768.0

class Generator(nn.Module):
    def __init__(self, mel_channel, num_layers: list = None):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel
        self.num_layers = num_layers
        
        self.generator = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mel_channel, 512, kernel_size=7, stride=1)),
            
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4)),
            
            ResStack(256, num_layers=self.num_layers[0]),
            
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4)),
            
            ResStack(128, num_layers=self.num_layers[1]),
            
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)),
            
            ResStack(64, num_layers=self.num_layers[2]),
            
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)),
            
            ResStack(32, num_layers=self.num_layers[3]),
            
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(32, 1, kernel_size=7, stride=1)),
            nn.Tanh(),
        )
    
    def forward(self, mel):
        mel = (mel + 5.0) / 5.0  # roughly normalize spectrogram
        return self.generator(mel)
    
    def eval(self, inference=False):
        super(Generator, self).eval()
        
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()
    
    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()
    
    def inference(self, mel):
        hop_length = 256
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(mel.device)
        mel = torch.cat((mel, zero), dim=2)
        
        audio = self.forward(mel)
        audio = audio.squeeze()  # collapse all dimension except time axis
        audio = audio[:-(hop_length * 10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
        audio = audio.short().cpu()
        
        return audio