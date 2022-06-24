import numpy as np
from scipy import stats


def latent_to_index(latents):
    latents_sizes = np.array([3, 6, 40, 32, 32])
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))
    return np.dot(latents, latents_bases).astype(int)


def binarize_xpos(latent):
    latent = latent >= 2
    return latent


def sample_latent(size=1):
    latents_sizes = np.array([3, 6, 40, 32, 32])
    samples = np.zeros((size, latents_sizes.size))
    for lat_i, lat_size in enumerate(latents_sizes):
        if lat_i == 3:
            b = [stats.bernoulli.rvs(p=0.92) if binarize_xpos(s) ==1 else stats.bernoulli.rvs(p=1-0.92) for s in samples[:, 0]]
            samples[:, lat_i] = [np.random.randint(lat_size//2) if s == 0 else np.random.randint(lat_size//2) + lat_size//2 for s in b]
        else:
            samples[:, lat_i] = np.random.randint(lat_size, size=size)

    return samples


def binarize(latent):
    latent[:, 1] = [1 if s >= 2 else 0 for s in latent[:,1]]
    latent[:, 2] = [1 if s >= 0.8 else 0 for s in latent[:,2]]
    latent[:, 3] = [1 if s >= 3.22214631 else 0 for s in latent[:,3]]
    latent[:, 4] = [1 if s >= 0.51612903 else 0 for s in latent[:,4]]
    latent[:, 5] = [1 if s >= 0.51612903 else 0 for s in latent[:,5]]
    return latent

### train, auditの分割をどうするのか
### 分割すると、unfairにサンプリングした時に参照できないインデックスがでてくる