import numpy as np


def latent_to_index(latents):
    latents_sizes = np.array([3, 6, 40, 32, 32])
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))
    return np.dot(latents, latents_bases).astype(int)


def sample_latent(size=1):
    latents_sizes = np.array([3, 6, 40, 32, 32])
    samples = np.zeros((size, latents_sizes.size))
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)

    return samples