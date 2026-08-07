"""TransformerNet — fast feedforward style transfer network (Johnson et al. 2016)."""

import torch
import torch.nn as nn


class TransformerNet(nn.Module):
    """Feedforward image transformation network."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(3, 32, 9, 1),
            ConvBlock(32, 64, 3, 2),
            ConvBlock(64, 128, 3, 2),
        )
        # Residual blocks
        self.residuals = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        # Decoder
        self.decoder = nn.Sequential(
            UpsampleBlock(128, 64, 3, 2),
            UpsampleBlock(64, 32, 3, 2),
            nn.Conv2d(32, 3, 9, 1, 4),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residuals(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, pad),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, scale):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="nearest"),
            nn.Conv2d(in_c, out_c, kernel, 1, pad),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.net(x)
