import os
from sklearn.model_selection import train_test_split
from pandas import read_csv
import numpy as np
import torchvision
import torch


class JurkatDataset:
    def __init__(self, root, transforms, test_size):
        self.root = root
        self.transforms = transforms
        self.img_path = os.path.join(self.root, "jurkat", "cell_cycle")
        self.annotations_path = os.path.join(self.root, "jurkat", "img.lst")
        self.annotations_columns = ['id', 'class', 'path']
        self.test_size = test_size

    def get_dataset(self):
        train_dataset = torchvision.datasets.ImageFolder(
            self.img_path, transform=self.transforms
        )

        annotations = read_csv(self.annotations_path, sep='\t', header=None)
        annotations.columns = self.annotations_columns

        train_idx, valid_idx = train_test_split(
            np.arange(annotations.shape[0]),
            test_size=self.test_size,
            shuffle=True,
            stratify=annotations['class'].tolist())
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        return train_dataset, train_sampler, valid_sampler
