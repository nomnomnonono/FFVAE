import numpy as np
import torch
from torch import nn


def get_decoder():
    return eval("Decoder")


class Decoder(nn.Module):
    def __init__(self, img_size, dataset, latent_dim=10):
        super(Decoder, self).__init__()

        hid_channels = 16
        kernel_size = 4
        hidden_dim = 64
        self.img_size = img_size
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        self.lin1 = nn.Linear(latent_dim, hidden_dim * 2)
        self.lin2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.lin3 = nn.Linear(hidden_dim * 2, np.product(self.reshape))

        cnn_kwargs = dict(stride=2, padding=1)
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(
                hid_channels, hid_channels, kernel_size, **cnn_kwargs
            )

        self.convT1 = nn.ConvTranspose2d(
            hid_channels, hid_channels // 2, kernel_size, **cnn_kwargs
        )
        self.convT2 = nn.ConvTranspose2d(
            hid_channels // 2, hid_channels // 2, kernel_size, **cnn_kwargs
        )
        self.convT3 = nn.ConvTranspose2d(
            hid_channels // 2, n_chan, kernel_size, **cnn_kwargs
        )

    def forward(self, z):
        batch_size = z.size(0)
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))  # dont used in dsprites
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))

        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        x = torch.sigmoid(self.convT3(x))
        return x
