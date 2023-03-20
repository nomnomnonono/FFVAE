"""
Module containing discriminator for FactorVAE.
"""
from disvae.utils.initialization import weights_init
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.z_dim = latent_dim
        self.hidden_units = 1000

        self.lin1 = nn.Linear(self.z_dim, self.hidden_units)
        self.lin2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin3 = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin4 = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin5 = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin6 = nn.Linear(self.hidden_units, 2)
        self.reset_parameters()

    def forward(self, z):
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.leaky_relu(self.lin3(z))
        z = self.leaky_relu(self.lin4(z))
        z = self.leaky_relu(self.lin5(z))
        z = self.lin6(z)
        return z

    def reset_parameters(self):
        self.apply(weights_init)
