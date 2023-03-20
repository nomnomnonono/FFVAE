"""
Module containing discriminator for FactorVAE.
"""
import torch
from disvae.utils.initialization import weights_init
from torch import nn


class MLP(nn.Module):
    def __init__(self, neg_slope=0.2, latent_dim=10, hidden_units=128):
        super(MLP, self).__init__()
        self.neg_slope = neg_slope
        self.leaky_relu = nn.LeakyReLU(self.neg_slope, True)
        self.z_dim = latent_dim
        self.hidden_units = hidden_units
        self.lin1 = nn.Linear(self.z_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, hidden_units)
        self.lin3 = nn.Linear(hidden_units, 1)

        self.reset_parameters()

    def forward(self, z, mode):
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.lin3(z)
        if mode == "test":
            z = nn.softmax(z, dim=1)
            return z
        elif mode == "train":
            prob, logit = torch.sigmoid(z), z
            return logit, prob

    def reset_parameters(self):
        self.apply(weights_init)
