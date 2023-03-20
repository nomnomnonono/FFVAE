import json
import os

import torch
from disvae.models.vae import init_specific_model

MODEL_FILENAME = "model.pt"
META_FILENAME = "specs.json"


def save_model(model, config, filename=MODEL_FILENAME):
    device = next(model.parameters()).device
    model.cpu()
    path_to_model = os.path.join(config.res_dir, filename)
    torch.save(model.state_dict(), path_to_model)
    model.to(device)

    save_metadata(dict(config), config.res_dir)


def load_metadata(directory, filename=META_FILENAME):
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata) as metadata_file:
        metadata = json.load(metadata_file)

    return metadata


def save_metadata(metadata, directory, filename=META_FILENAME):
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata, "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)


def load_model(directory, dataset, n_sens, is_gpu=True, filename=MODEL_FILENAME):
    device = torch.device("cuda" if torch.cuda.is_available() and is_gpu else "cpu")
    path_to_model = os.path.join(directory, filename)
    metadata = load_metadata(directory)
    img_size = metadata["img_size"]
    latent_dim = metadata["latent_dim"]
    model = _get_model(img_size, latent_dim, dataset, n_sens, device, path_to_model)
    return model


def _get_model(img_size, latent_dim, dataset, n_sens, device, path_to_model):
    model = init_specific_model(img_size, latent_dim, dataset, n_sens).to(device)
    model.load_state_dict(torch.load(path_to_model), strict=False)
    model.eval()
    return model
