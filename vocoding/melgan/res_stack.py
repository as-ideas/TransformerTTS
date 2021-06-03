"""
CODE ADAPTED FROM
https://github.com/seungwonpark/melgan
"""

import torch.nn as nn


class ResStack(nn.Module):
    def __init__(self, channel, num_layers=4):
        super(ResStack, self).__init__()
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3 ** i),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=3 ** i)),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )
            for i in range(num_layers)
        ])
        
        self.shortcuts = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x
    
    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)
