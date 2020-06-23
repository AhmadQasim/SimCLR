import os
import torchvision


class MatekDataset:
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.train_path = os.path.join(self.root, "matek", "train")
        self.test_path = os.path.join(self.root, "matek", "test")

    def get_dataset(self):
        train_dataset = torchvision.datasets.ImageFolder(
            self.train_path, transform=self.transforms
        )

        test_dataset = torchvision.datasets.ImageFolder(
            self.test_path, transform=self.transforms
        )

        return train_dataset, test_dataset
