import os
import random
import shutil

import numpy as np
import torch


def create_safe_directory(directory):
    if os.path.exists(directory):
        shutil.make_archive(directory, "zip", directory)
        shutil.rmtree(directory)
    os.makedirs(directory)


def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def get_device(is_gpu=True):
    return torch.device("cuda" if torch.cuda.is_available() and is_gpu else "cpu")
