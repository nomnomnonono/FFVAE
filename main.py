import argparse

from disvae.models.losses import FFVAELoss
from disvae.models.mlp import MLP
from disvae.models.vae import init_specific_model
from disvae.training import MLPTrainer, Trainer
from disvae.utils.modelIO import save_model
from omegaconf import OmegaConf
from torch import optim
from utils.datasets import get_dataloaders, get_img_size
from utils.helpers import create_safe_directory, get_device, set_seed


def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="File path for config file.",
    )
    args = parser.parse_args()
    return args


def main(args):
    config = OmegaConf.load(args.config)
    set_seed(config.seed)
    device = get_device(is_gpu=not config.no_cuda)

    # train vae
    create_safe_directory(config.res_dir)

    train_loader = get_dataloaders("train", config)
    img_size = get_img_size(config.dataset)

    model = init_specific_model(
        img_size, config.latent_dim, config.dataset, config.n_sens
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    loss = FFVAELoss(device, config.alpha, config.gamma, config.latent_dim)
    trainer = Trainer(
        model,
        optimizer,
        loss,
        device=device,
        save_dir=config.res_dir,
    )
    trainer(
        train_loader,
        epochs=config.vae_epochs,
        checkpoint_every=config.checkpoint_every,
    )

    save_model(trainer.model, config)

    # train mlp
    train_loader = get_dataloaders("mlp", config)
    val_loader = get_dataloaders("val", config)
    test_loader = get_dataloaders("test", config)

    mlp = MLP(latent_dim=config.latent_dim).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=config.mlp_lr)

    mlp_trainer = MLPTrainer(
        mlp,
        model,
        optimizer,
        config.target_sens,
        config.y,
        config.mlp_epochs,
        config.mlp_lr,
        device=device,
        save_dir=config.res_dir,
    )

    mlp_trainer(train_loader, val_loader, test_loader, epochs=config.mlp_epochs)


if __name__ == "__main__":
    args = argparser()
    main(args)
