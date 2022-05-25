# coding: utf-8

from torch import nn


class PixelShuffle(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()

        scale_factor = 2
        base_channels = 32
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 2 * base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * base_channels, 2 * base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * base_channels, 2 * scale_factor, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, x):
        return self.model(x)
