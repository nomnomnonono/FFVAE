import glob
import logging
import os
from collections import defaultdict

import torch
from disvae.utils.modelIO import save_model
from torch import nn
from tqdm import trange

DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_LOSSES_LOGFILE = "train_losses.log"


class LossesLogger(object):
    def __init__(self, file_path_name):
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer):
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, sum(v) / len(v)])
            self.logger.debug(log_string)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss,
        device,
        save_dir,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.losses_logger = LossesLogger(
            os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE)
        )

    def __call__(self, data_loader, epochs, checkpoint_every):
        self.model.train()
        print("---------- Encoder Train ----------")

        for epoch in range(epochs):
            storer = defaultdict(list)
            mean_epoch_loss = self._train_epoch(data_loader, storer, epoch)
            self.losses_logger.log(epoch, storer)
            print(f"Epoch {epoch+1}: {mean_epoch_loss:.2f}")

            if (epoch + 1) % checkpoint_every == 0:
                save_model(
                    self.model,
                    self.save_dir,
                    filename="model-{}.pt".format(epoch + 1),
                )
        self.model.eval()

    def _train_epoch(self, data_loader, storer, epoch):
        epoch_loss = 0.0
        kwargs = dict(
            desc="Epoch {}".format(epoch + 1),
            leave=False,
        )
        with trange(len(data_loader), **kwargs) as t:
            for _, (data, sens, _) in enumerate(data_loader):
                data = data.float()
                iter_loss = self._train_iteration(data, sens, storer)
                epoch_loss += iter_loss

                t.set_postfix(loss=iter_loss)
                t.update()

        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss

    def _train_iteration(self, data, sens, storer):
        data = data.to(self.device, non_blocking=True)
        recon_batch, latent_dist, latent_sample = self.model(data)
        loss = self.loss(
            data,
            sens,
            self.optimizer,
            recon_batch,
            latent_dist,
            self.model.training,
            storer,
            self.model,
            latent_sample=latent_sample,
        )
        return loss.item()


class MLPTrainer:
    def __init__(
        self,
        model,
        vae,
        optimizer,
        target_sens,
        y,
        epoch,
        mlp_lr,
        device,
        save_dir,
    ):

        self.device = device
        self.model = model.to(self.device)
        self.vae = vae.to(self.device)
        self.optimizer = optimizer
        self.target_sens = target_sens
        self.y = y
        self.save_dir = save_dir
        self.logger = LossesLogger(
            os.path.join(
                self.save_dir,
                "{}-{}-{}-mlp_train_losses-{}.log".format(
                    mlp_lr, epoch, self.y, self.target_sens
                ),
            )
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.min_val = None
        self.count = 0
        self.flag = False

    def __call__(self, data_loader, val_loader, test_loader, epochs=10):
        self.model.train()
        self.vae.eval()
        print("---------- MLP Train ----------")

        for epoch in range(epochs):
            train_storer = defaultdict(list)
            val_storer = defaultdict(list)
            test_storer = defaultdict(list)
            mean_epoch_loss, mean_epoch_acc, mean_epoch_dp = self._mlp_train_epoch(
                data_loader,
                val_loader,
                test_loader,
                train_storer,
                val_storer,
                test_storer,
                epoch,
            )
            self.logger.log(epoch, train_storer)
            self.logger.log(epoch, val_storer)
            self.logger.log(epoch, test_storer)
            print(
                f"Epoch {epoch+1}: Acc.{mean_epoch_acc*100:.2f}, DP.{mean_epoch_dp:.3f}"
            )
            self.model.cpu()
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_dir, "mlp-{}.pt".format(epoch)),
            )

            if self.flag:
                os.rename(
                    os.path.join(self.save_dir, "mlp-{}.pt".format(epoch - 4)),
                    os.path.join(
                        self.save_dir, "{}-mlp_{}.pt".format(self.y, self.target_sens)
                    ),
                )
                files = glob.glob(os.path.join(self.save_dir, "mlp-*.pt"))
                for file in files:
                    os.remove(file)
                break

            self.model.to(self.device)

        self.model.eval()

    def _mlp_train_epoch(
        self,
        data_loader,
        val_loader,
        test_loader,
        train_storer,
        val_storer,
        test_storer,
        epoch,
    ):
        epoch_loss, epoch_acc, epoch_dp = 0.0, 0.0, 0.0
        kwargs = dict(
            desc="Epoch {}".format(epoch + 1),
            leave=False,
        )
        with trange(len(data_loader), **kwargs) as t:
            for _, (data, sens, label) in enumerate(data_loader):
                data, sens, label = (
                    data.to(self.device),
                    sens.to(self.device),
                    label.to(self.device),
                )
                data = data.float()
                latent = self.vae.sample_latent(data).detach()
                latent[:, self.target_sens] = torch.randn_like(
                    latent[:, self.target_sens]
                )
                logit, prob = self.model(latent, mode="train")
                loss = self.loss(logit.view(-1), label)
                acc = sum((prob.view(-1) > 0.5) == label).float().item() / len(label)
                pred = prob.view(-1) > 0.5
                dp = calc_dp(pred, sens, self.target_sens)
                if train_storer is not None:
                    train_storer["clf_train"].append(loss.item())
                    train_storer["acc_train"].append(acc)
                    train_storer["dp_train"].append(dp)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc
                epoch_dp += dp

                t.set_postfix(loss=loss.item())
                t.update()

        val_loss, val_acc, val_dp = 0.0, 0.0, 0.0
        with torch.no_grad():
            with trange(len(val_loader), **kwargs) as t:
                for _, (data, sens, label) in enumerate(val_loader):
                    data, sens, label = (
                        data.to(self.device),
                        sens.to(self.device),
                        label.to(self.device),
                    )
                    data = data.float()
                    latent = self.vae.sample_latent(data).detach()
                    self.model.eval()
                    logit, prob = self.model(latent, mode="train")
                    loss = self.loss(logit.view(-1), label)
                    pred = prob.view(-1) > 0.5
                    dp = calc_dp(pred, sens, self.target_sens)
                    if val_storer is not None:
                        val_storer["clf_val"].append(loss.item())
                        val_storer["acc_val"].append(acc)
                        val_storer["dp_val"].append(dp)
                    val_loss += loss.item()
                    val_acc += acc
                    val_dp += dp

                    t.set_postfix(loss=loss.item())
                    t.update()
                val_loss = val_loss / len(val_loader)
                val_acc = val_acc / len(val_loader)
                val_dp = val_dp / len(val_loader)

                if self.min_val is None:
                    self.min_val = val_loss
                else:
                    if val_loss < self.min_val:
                        self.min_val = val_loss
                        self.count = 0
                    else:
                        self.count += 1
                if self.count > 4:
                    self.flag = True

        test_loss, test_acc, test_dp = 0.0, 0.0, 0.0
        with torch.no_grad():
            with trange(len(test_loader), **kwargs) as t:
                for _, (data, sens, label) in enumerate(test_loader):
                    data, sens, label = (
                        data.to(self.device),
                        sens.to(self.device),
                        label.to(self.device),
                    )
                    data = data.float()
                    latent = self.vae.sample_latent(data).detach()
                    latent[:, self.target_sens] = torch.randn_like(
                        latent[:, self.target_sens]
                    )
                    self.model.eval()
                    logit, prob = self.model(latent, mode="train")
                    loss = self.loss(logit.view(-1), label)
                    acc = sum((prob.view(-1) > 0.5) == label).float().item() / len(
                        label
                    )
                    pred = prob.view(-1) > 0.5
                    dp = calc_dp(pred, sens, self.target_sens)
                    if test_storer is not None:
                        test_storer["clf_test"].append(loss.item())
                        test_storer["acc_test"].append(acc)
                        test_storer["dp_test"].append(dp)
                    test_loss += loss.item()
                    test_acc += acc
                    test_dp += dp

                    t.set_postfix(loss=loss.item())
                    t.update()
                test_loss = test_loss / len(test_loader)
                test_acc = test_acc / len(test_loader)
                test_dp = test_dp / len(test_loader)

        self.model.train()
        return test_loss, test_acc, test_dp


def calc_dp(pred, sens, target_sens):
    a1, a0, y1a1, y1a0 = 0.0, 0.0, 0.0, 0.0
    for i in range(len(pred)):
        if pred[i] == 1:
            if sens[i, target_sens] == 1:
                y1a1 += 1
                a1 += 1
            else:
                y1a0 += 1
                a0 += 1
        else:
            if sens[i, target_sens] == 1:
                a1 += 1
            else:
                a0 += 1
    y1a1, y1a0 = y1a1 / a1, y1a0 / a0
    return abs(y1a1 - y1a0)
