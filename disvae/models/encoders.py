import numpy as np
import torch
from torch import nn


def get_encoder():
    return eval("Encoder")


class Encoder(nn.Module):
    def __init__(self, img_size, dataset, latent_dim=10):
        super(Encoder, self).__init__()

        hid_channels = 16
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = int(latent_dim)
        self.img_size = img_size
        self.reshape = (hid_channels * 2, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(
            hid_channels, hid_channels * 2, kernel_size, **cnn_kwargs
        )

        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(
                hid_channels * 2, hid_channels * 2, kernel_size, **cnn_kwargs
            )

        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))

        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))  # not used in dsprites
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar
