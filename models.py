import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, channels):
        super(Generator, self).__init__()

        def block(in_channels, out_channels):
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.Upsample(scale_factor=2),
                nn.LeakyReLU(0.2, True),
            ]

        self.init_size = 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))

        self.model = nn.Sequential(
            *block(512, 256),
            *block(256, 128),
            *block(128, 64),
            *block(64, 32),

            nn.Conv2d(32, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        out = self.model(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()

        def block(in_channels, out_channels):
            return [
                nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True),
            ]

        self.main = nn.Sequential(
            *block(channels, 16),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
            *block(128, 256),

            nn.Conv2d(256, 1, 4, 1, 0, bias=True),
        )

    def forward(self, img):
        out = self.main(img)
        out = torch.flatten(out, 1)

        return out