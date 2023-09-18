import math

from torch.nn import functional as F
from torch import nn

from generator import InputBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=s, bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.sequential(x)


class Discriminator(nn.Module):
    def __init__(self, upscale_factor=2) -> None:
        super().__init__()
        self.input = InputBlock(gen=False)
        out_channels = 64
        in_channels = 64
        self.__conv_blocks = []
        strides = [1, 2]
        for i in range(7):
            self.__conv_blocks.append(
                ConvBlock(in_channels=in_channels, out_channels=out_channels,
                          s=strides[0])
            )
            strides = list(reversed(strides))
            in_channels = out_channels
            out_channels *= 2 if i % 2 == 0 else 1
        self.conv_blocks_seq = nn.Sequential(*self.__conv_blocks)

        self.classifier = nn.Sequential(
            nn.Linear(512*(upscale_factor*3)*(upscale_factor*3), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv_blocks_seq(x)
        x = x.view(-1, math.prod(x.shape[1:]))
        x = self.classifier(x)
        return F.sigmoid(x)
