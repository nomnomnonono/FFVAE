import torch
from disvae.utils.initialization import weights_init
from torch import nn

from .decoders import get_decoder
from .encoders import get_encoder


def init_specific_model(img_size, latent_dim, dataset, n_sens):
    encoder = get_encoder()
    decoder = get_decoder()
    model = VAE(img_size, encoder, decoder, latent_dim, dataset, n_sens)
    return model


class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim, dataset, n_sens):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, dataset, self.latent_dim)
        self.decoder = decoder(img_size, dataset, self.latent_dim)
        self.n_sens = n_sens
        self.sens_idx = list(range(self.n_sens))
        self.nonsens_idx = list(range(self.n_sens + 1, self.latent_dim))
        self.reset_parameters()

    def reparameterize(self, _mean, _logvar):
        if self.training:
            mean = _mean[:, self.nonsens_idx]
            logvar = _logvar[:, self.nonsens_idx]
            zb = torch.zeros_like(_mean)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + std * eps
            b = _mean[:, self.sens_idx]
            zb[:, self.sens_idx] = b
            zb[:, self.nonsens_idx] = z
            return zb
        else:
            return _mean

    def forward(self, x):
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample
