from torch.nn import functional as F
from torch import nn


class InputBlock(nn.Module):
    def __init__(self, gen=True) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=9, stride=1,
                padding=4
            ),
            nn.PReLU(num_parameters=64) if gen else nn.LeakyReLU(
                0.2, inplace=True)
        )

    def forward(self, x):
        return self.sequential(x)


class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(num_parameters=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      bias=False, padding=1),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        out = self.sequential(x)
        return out + x


class UpsampleBlock(nn.Module):
    def __init__(self, upscale_factor) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64 * upscale_factor ** 2,
                      kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            # c(out) = c(in) / upscale_factor ** 2
            # = (256 / `upscale_factor` ** 2)
            # = 64
            # H(out) = H(in) * upscale_factor
            # = (24 * `upscale_factor`)
            # = 48
            # W(out) = W(in) * upscale_factor
            # = (24 * `upscale_factor`)
            # = 48
            # (N, 64, 48, 48)
            nn.PReLU(64)
        )

    def forward(self, x):
        return self.sequential(x)


class Generator(nn.Module):
    def __init__(self, upscale_factor=2) -> None:
        super().__init__()
        self.input_block = InputBlock()
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock() for _ in range(5)])
        self.upsample_blocks = nn.Sequential(
            *[UpsampleBlock(upscale_factor) for _ in range(1)])
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.last_conv = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4
        )

    def forward(self, x):
        initial = self.input_block(x)
        out = self.residual_blocks(initial)
        out = self.sequential(out) + initial
        out = self.upsample_blocks(out)
        return F.tanh(self.last_conv(out))
