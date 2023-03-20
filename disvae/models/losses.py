import abc

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

from .discriminator import Discriminator


class BaseLoss(abc.ABC):
    def __init__(self, record_loss_every=50, steps_anneal=0):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        pass

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if not is_train or self.n_train_steps % self.record_loss_every == 1:
            storer = storer
        else:
            storer = None

        return storer


class FFVAELoss(BaseLoss):
    def __init__(
        self,
        device,
        alpha,
        gamma,
        latent_dim,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.discriminator = Discriminator(latent_dim).to(self.device)
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=5e-5, betas=(0.5, 0.9)
        )

    def __call__(
        self,
        data,
        sens,
        optimizer,
        recon_batch,
        latent_dist,
        is_train,
        storer,
        model,
        latent_sample=None,
    ):
        storer = self._pre_call(is_train, storer)
        batch_size, latent_dim = latent_sample.shape

        rec_loss = _reconstruction_loss(data, recon_batch, storer=storer)

        n_sens = sens.shape[1]
        sens_idx = list(range(n_sens))
        nonsens_idx = list(range(n_sens + 1, 50))
        _mu, _logvar = latent_dist
        b_logits = _mu[:, sens_idx]

        clf_loss = [
            nn.BCEWithLogitsLoss()(_b_logit.to(self.device), _a_sens.to(self.device))
            for _b_logit, _a_sens in zip(
                b_logits.squeeze().t(),
                sens.type(torch.FloatTensor).squeeze().t(),
            )
        ]
        clf_loss_mean = torch.stack(clf_loss).mean()

        d_z = self.discriminator(latent_sample)
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()

        _mu, _logvar = latent_dist
        mu = _mu[:, nonsens_idx]
        logvar = _logvar[:, nonsens_idx]
        std = (logvar / 2).exp()
        q_zIx = torch.distributions.Normal(mu, std)
        p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        dw_kl_loss = torch.distributions.kl_divergence(q_zIx, p_z).sum(1).mean()

        anneal_reg = (
            linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
            if is_train
            else 1
        )

        loss = rec_loss + (
            self.alpha * clf_loss_mean + self.gamma * tc_loss + anneal_reg * dw_kl_loss
        )
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        if storer is not None:
            storer["loss"].append(loss.item())
            storer["clf_loss"].append(clf_loss_mean.item())
            storer["tc_loss"].append(tc_loss.item())
            storer["dw_kl_loss"].append(dw_kl_loss.item())

        z_fake = torch.zeros_like(latent_sample)
        for i in range(n_sens):
            z_fake[:, i] = latent_sample[:, i][torch.randperm(latent_sample.shape[0])]
        z_fake[:, n_sens:] = latent_sample[:, n_sens:][
            torch.randperm(latent_sample.shape[0])
        ]
        z_fake = z_fake.to(self.device).detach()
        d_z_perm = self.discriminator(z_fake)

        ones = torch.ones(latent_sample.shape[0], dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones)
        d_tc_loss = 0.5 * (
            F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_perm, ones)
        )

        self.optimizer_d.zero_grad()
        d_tc_loss.backward()

        torch.nn.utils.clip_grad_norm_(list(self.discriminator.parameters()), 5.0)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), 5.0)
        optimizer.step()
        self.optimizer_d.step()

        return loss


def _reconstruction_loss(data, recon_data, storer=None):
    batch_size, n_chan, height, width = recon_data.size()
    loss = F.binary_cross_entropy(recon_data, data, reduction="sum")

    loss = loss / batch_size

    if storer is not None:
        storer["recon_loss"].append(loss.item())

    return loss


def linear_annealing(init, fin, step, annealing_steps):
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed
