from cgi import test
import imageio
import logging
import os
from timeit import default_timer
from collections import defaultdict

from tqdm import trange
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from disvae.utils.modelIO import save_model
from disvae.utils.dsprites import *
from disvae.models.losses import DemographicParityLoss


DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_LOSSES_LOGFILE = "train_losses.log"


class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    optimizer: torch.optim.Optimizer

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    gif_visualizer : viz.Visualizer, optional
        Gif Visualizer that should return samples at every epochs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, optimizer, loss_f, dataset,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 gif_visualizer=None,
                 is_progress_bar=True):

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.dataset = dataset
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.losses_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        self.gif_visualizer = gif_visualizer
        self.logger.info("Training Device: {}".format(self.device))

    def __call__(self, data_loader,
                 epochs=10,
                 iters=3*10**5,
                 checkpoint_every=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()

        if self.dataset == "dsprites":
            dataset_zip = np.load(os.path.join(DIR, "../data/dsprites/dsprite_train.npz"))
            imgs = dataset_zip["imgs"]
            latents_values = dataset_zip["latents_values"]
            # transforms = transforms.Compose([transforms.ToTensor()])
            storer = defaultdict(list)
            for iter in range(iters):
                latents_sampled = sample_latent(size=64)
                indices_sampled = latent_to_index(latents_sampled)
                imgs_sampled = imgs[indices_sampled]
                samples = torch.from_numpy(imgs_sampled.astype(np.float32))
                samples = samples.view(64, 1, 64, 64)
                sens = torch.from_numpy(binarize(latents_values[indices_sampled])[:, [1, 2]])
                iter_loss = self._train_iteration(samples, sens, storer)
                if (iter+1) % 10000 == 0:
                    self.logger.info('Iter: {} Average loss per image: {:.2f}'.format(iter+ 1, iter_loss))
                    self.losses_logger.log(iter+1, storer)

                    if self.gif_visualizer is not None:
                        self.gif_visualizer()
                    storer = defaultdict(list)
                    save_model(self.model, self.save_dir, filename="model-{}.pt".format(iter+1))
        else:
            for epoch in range(epochs):
                storer = defaultdict(list)
                mean_epoch_loss = self._train_epoch(data_loader, storer, epoch)
                self.logger.info('Epoch: {} Average loss per image: {:.2f}'.format(epoch + 1, mean_epoch_loss))
                self.losses_logger.log(epoch, storer)

                if self.gif_visualizer is not None:
                    self.gif_visualizer()

                if (epoch + 1) % checkpoint_every == 0:
                    save_model(self.model, self.save_dir, filename="model-{}.pt".format(epoch+1))

        if self.gif_visualizer is not None:
            self.gif_visualizer.save_reset()

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        self.logger.info('Finished training after {:.1f} min.'.format(delta_time))

    def _train_epoch(self, data_loader, storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=not self.is_progress_bar)
        with trange(len(data_loader), **kwargs) as t:
            for i, (data, sens, _) in enumerate(data_loader):
                data = data.float()
                iter_loss = self._train_iteration(data, sens, storer)
                epoch_loss += iter_loss

                t.set_postfix(loss=iter_loss)
                t.update()

        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss

    def _train_iteration(self, data, sens, storer):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        batch_size, channel, height, width = data.size()
        data = data.to(self.device)

        try:
            recon_batch, latent_dist, latent_sample = self.model(data)
            loss  = self.loss_f(data, sens, self.optimizer, recon_batch, latent_dist, self.model.training,
                               storer, latent_sample=latent_sample)

        except ValueError:
            # for losses that use multiple optimizers (e.g. Factor)
            loss = self.loss_f.call_optimize(data, self.model, self.optimizer, storer)

        return loss.item()


class LossesLogger(object):
    """Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """

    def __init__(self, file_path_name):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer):
        """Write to the log file """
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, mean(v)])
            self.logger.debug(log_string)


# HELPERS
def mean(l):
    """Compute the mean of a list"""
    return sum(l) / len(l)


class MLPTrainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    optimizer: torch.optim.Optimizer

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    gif_visualizer : viz.Visualizer, optional
        Gif Visualizer that should return samples at every epochs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, vae, optimizer, target_sens,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 is_progress_bar=True):

        self.device = device
        self.model = model.to(self.device)
        self.vae = vae.to(self.device)
        self.optimizer = optimizer
        self.target_sens = target_sens
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.losses_logger = LossesLogger(os.path.join(self.save_dir, "mlp_train_losses-{}.log".format(self.target_sens)))
        self.test_logger = LossesLogger(os.path.join(self.save_dir, "mlp_test_losses-{}.log".format(self.target_sens)))
        self.logger.info("Training Device: {}".format(self.device))
        self.dp = DemographicParityLoss()
        self.loss = nn.BCEWithLogitsLoss()
        self.min_val = None
        self.count = 0
        self.flag = False

    def __call__(self, data_loader, test_loader,
                 epochs=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()
        self.vae.eval()

        for epoch in range(epochs):
            train_storer = defaultdict(list)
            test_storer = defaultdict(list)
            mean_epoch_loss, mean_epoch_acc, mean_epoch_dp = self._mlp_train_epoch(data_loader, test_loader, train_storer, test_storer, epoch)
            self.logger.info('Epoch: {} Average loss per image: {:.4f}. acc: {:.4f}. dp: {:.4f}'.format(
                epoch + 1, mean_epoch_loss, mean_epoch_acc, mean_epoch_dp))
            self.losses_logger.log(epoch, train_storer)
            self.test_logger.log(epoch, test_storer)
            self.model.cpu()
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "mlp-{}.pt".format(epoch)))
            self.model.to(self.device)
            if self.flag:
                self.model.cpu()
                os.rename(os.path.join(self.save_dir, "mlp-{}.pt".format(epoch-5)), os.path.join(self.save_dir, "mlp_{}.pt".format(self.target_sens)))
                break

            #save_model(self.model, self.save_dir, filename="mlp-{}.pt".format(epoch+1))

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        self.logger.info('Finished training after {:.1f} min.'.format(delta_time))

    def _mlp_train_epoch(self, data_loader, test_loader, train_storer, test_storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_loss = 0.
        epoch_acc = 0.
        epoch_dp = 0.
        kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=not self.is_progress_bar)
        with trange(len(data_loader), **kwargs) as t:
            for _, (data, sens, label) in enumerate(data_loader):
                data, sens, label = (
                    data.to(self.device),
                    sens.to(self.device),
                    label.to(self.device)
                )
                data = data.float()
                latent = self.vae.sample_latent(data).detach()
                latent[:, self.target_sens] = torch.randn_like(latent[:, self.target_sens])
                logit, prob = self.model(latent, mode="train")
                loss = self.loss(logit.view(-1), label)
                acc = sum((prob.view(-1) > 0.5) == label).float().item() / len(label)
                dp = self.dp(data, logit, sens[:, self.target_sens])
                if train_storer is not None:
                    train_storer['clf'].append(loss.item())
                    # Acc, DP
                    train_storer['acc'].append(acc)
                    train_storer['dp'].append(dp.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc
                epoch_dp += dp.item()

                t.set_postfix(loss=loss.item())
                t.update()
        
        test_loss = 0.
        test_acc = 0.
        test_dp = 0.
        with trange(len(test_loader), **kwargs) as t:
            for _, (data, sens, label) in enumerate(test_loader):
                data, sens, label = (
                    data.to(self.device),
                    sens.to(self.device),
                    label.to(self.device)
                )
                data = data.float()
                latent = self.vae.sample_latent(data).detach()
                latent[:, self.target_sens] = torch.randn_like(latent[:, self.target_sens])
                self.model.eval()
                logit, prob = self.model(latent, mode="train")
                loss = self.loss(logit.view(-1), label)
                acc = sum((prob.view(-1) > 0.5) == label).float().item() / len(label)
                pred = prob.view(-1) > 0.5
                y1a1 = 0.
                y1a0 = 0.
                a1 = 0.
                a0 = 0.
                for i in range(len(pred)):
                    if pred[i] == 1:
                        if sens[i, self.target_sens] == 1:
                            y1a1 += 1
                            a1 += 1
                        else:
                            y1a0 += 1
                            a0 += 1
                    else:
                        if sens[i, self.target_sens] == 1:
                            a1 += 1
                        else:
                            a0 += 1
                y1a1 /= a1
                y1a0 /= a0
                dp = abs(a1 - a0)
                if test_storer is not None:
                    test_storer['clf'].append(loss.item())
                    # Acc, DP
                    test_storer['acc'].append(acc)
                    test_storer['dp'].append(dp)
                test_loss += loss.item()
                test_acc += acc
                test_dp += dp

                t.set_postfix(loss=loss.item())
                t.update()
            test_loss = test_loss / len(test_loader)
            test_acc = test_acc / len(test_loader)
            test_dp = test_dp / len(test_loader)
            
            if self.min_val is None:
                self.min_val = test_loss
                self.count += 1
            else:
                if test_loss < self.min_val:
                    self.min_val = test_loss
                    self.count = 0
                else:
                    self.count += 1
            if self.count > 4:
                self.flag = True

            self.model.train()


        mean_epoch_loss = epoch_loss / len(data_loader)
        mean_epoch_acc = epoch_acc / len(data_loader)
        mean_epoch_dp = epoch_dp / len(data_loader)
        #return mean_epoch_loss, mean_epoch_acc, mean_epoch_dp
        return test_loss, test_acc, test_dp
