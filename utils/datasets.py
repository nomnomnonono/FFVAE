import glob
import os

import numpy as np
import pandas as pd
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DICT = {
    "celeba": "CelebA",
    "utkface": "UTKFace",
}
DATASETS = list(DATASETS_DICT.keys())


def get_dataset(dataset):
    dataset = dataset.lower()
    return eval(DATASETS_DICT[dataset])


def get_img_size(dataset):
    return get_dataset(dataset).img_size


def get_dataloaders(which_set, config):
    Dataset = get_dataset(config.dataset)
    dataset = Dataset(which_set, config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )


class CelebA(Dataset):
    img_size = [3, 64, 64]

    def __init__(self, which_set, config):
        self.y = config.y
        self.s = config.s
        self.transforms = transforms.Compose([transforms.ToTensor()])

        if which_set == "train":
            self.imgs = glob.glob(os.path.join(config.root, "train") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(config.root, "train-label.txt"))
            self.labels = pd.read_csv(self.labels).replace(-1, 0)
        elif which_set == "mlp":
            self.imgs = glob.glob(os.path.join(config.root, "mlp") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(config.root, "mlp-label.txt"))
            self.labels = pd.read_csv(self.labels).replace(-1, 0)
        elif which_set == "val":
            self.imgs = glob.glob(os.path.join(config.root, "val") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(config.root, "label.txt"))
            self.labels = pd.read_csv(self.labels, sep=" ").replace(-1, 0)[162770:]
        elif which_set == "test":
            self.imgs = glob.glob(os.path.join(config.root, "test") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(config.root, "label.txt"))
            self.labels = pd.read_csv(self.labels, sep=" ").replace(-1, 0)[182637:]
        else:
            raise ValueError("Unknown Dataset Type")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = imread(img_path)
        img = self.transforms(img)
        sens = np.array([self.labels.iloc[idx][self.s]], dtype="float")
        label = np.array(self.labels.iloc[idx][self.y], dtype="float")
        return img, sens, label


class UTKFace(Dataset):
    img_size = [3, 64, 64]

    def __init__(self, which_set, config):
        self.img_size = config.img_size
        self.which_set = which_set
        self.y = config.y
        self.s = config.s
        self.transforms = transforms.Compose([transforms.ToTensor()])

        if which_set == "train":
            self.imgs = glob.glob(os.path.join(config.root, "train") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(config.root, "train-label.csv"))
            self.labels = pd.read_csv(self.labels)
        elif which_set == "mlp":
            self.imgs = glob.glob(os.path.join(config.root, "mlp") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(config.root, "mlp-label.csv"))
            self.labels = pd.read_csv(self.labels)
        elif which_set == "val":
            self.imgs = glob.glob(os.path.join(config.root, "val") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(config.root, "label.csv"))
            self.labels = pd.read_csv(self.labels)[18964:]
        elif which_set == "test":
            self.imgs = glob.glob(os.path.join(config.root, "test") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(config.root, "label.csv"))
            self.labels = pd.read_csv(self.labels)[21334:]
        else:
            raise ValueError("Unknown Dataset Type")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = imread(img_path)
        img = self.transforms(img)
        sens = np.array([self.labels.iloc[idx][self.s]], dtype="float")
        label = np.array(self.labels.iloc[idx][self.y], dtype="float")
        return img, sens, label
